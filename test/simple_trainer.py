import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import glob

from factory import load_leela_model, save_leela_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimpleTrainingConfig:
    """Simple configuration for PIMCTS training"""
    # Data
    training_data_path: str = "pimcts_training_data"
    validation_split: float = 0.1
    
    # Model paths (matching your selfplay script)
    model_path: str = "eval_small.safetensors"
    config_path: str = "eval_small.json"
    output_model_path: str = "eval_small_trained.safetensors"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 5
    
    # Loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    
    # Optimization
    gradient_clip_norm: float = 1.0
    use_mixed_precision: bool = True
    
    # Hardware
    device: str = "auto"
    num_workers: int = 4


class SimpleDataset(Dataset):
    """Simple dataset for PIMCTS training data"""
    
    def __init__(self, training_data_path: str):
        """Load all .npz files from the training data directory"""
        self.data_path = Path(training_data_path)
        
        # Find all .npz files
        data_files = list(self.data_path.glob("**/*.npz"))
        if not data_files:
            raise FileNotFoundError(f"No .npz files found in {self.data_path}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Load and concatenate all data
        all_positions = []
        all_policies = []
        all_values = []
        
        for data_file in data_files:
            logger.info(f"Loading {data_file}")
            try:
                data = np.load(data_file)
                
                positions = data['position_encodings']
                policies = data['policy_targets']
                values = data['value_targets']
                
                logger.info(f"  - {len(positions)} samples")
                
                all_positions.append(positions)
                all_policies.append(policies)
                all_values.append(values)
                
            except Exception as e:
                logger.warning(f"  - Failed to load {data_file}: {e}")
                continue
        
        if not all_positions:
            raise ValueError("No valid data files found")
        
        # Concatenate all data
        self.positions = np.concatenate(all_positions, axis=0)
        self.policies = np.concatenate(all_policies, axis=0)
        self.values = np.concatenate(all_values, axis=0)
        
        logger.info(f"Total dataset: {len(self.positions)} samples")
        logger.info(f"Position shape: {self.positions.shape}")
        logger.info(f"Policy shape: {self.policies.shape}")
        logger.info(f"Value shape: {self.values.shape}")
        
        # Handle position encoding shape conversion
        # Expected model input: [batch, 90, 167] for transformer
        # Your data might be: [batch, 1, 90, 167] or already [batch, 90, 167]
        
        if self.positions.ndim == 4 and self.positions.shape[1] == 1:  # [N, 1, 90, 167]
            # Remove singleton dimension
            self.positions = self.positions.squeeze(1)  # [N, 90, 167]
            logger.info("Removed singleton dimension from positions")
            
        elif self.positions.ndim == 3 and self.positions.shape[-1] == 167:  # [N, 90, 167]
            # Already correct shape
            logger.info("Positions already in correct shape")
            
        else:
            raise ValueError(f"Unexpected position shape: {self.positions.shape}. "
                           f"Expected [N, 90, 167] or [N, 1, 90, 167]")
        
        logger.info(f"Final position shape: {self.positions.shape}")
        
        # Convert to tensors
        self.positions = torch.tensor(self.positions, dtype=torch.float32)
        self.policies = torch.tensor(self.policies, dtype=torch.float32)
        self.values = torch.tensor(self.values, dtype=torch.float32)

        # -- FIXED --
        # IMPORTANT: Convert WDL to LDW to match C++ MCTS expectation
        # Self-play produces WDL [win_prob, draw_prob, loss_prob]
        # C++ model expects LDW [loss_prob, draw_prob, win_prob]
        # So we need to flip the first and last elements
        # logger.info("Converting WDL targets to LDW format for C++ compatibility")
        # self.values = self.values[:, [2, 1, 0]]  # [W, D, L] -> [L, D, W]
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.positions[idx], self.policies[idx], self.values[idx]


class CombinedLoss(nn.Module):
    """AlphaZero-style loss function adapted for WDL (without manual L2 reg)"""
    
    def __init__(self, policy_weight: float = 1.0, value_weight: float = 1.0):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        
        # AlphaZero uses cross-entropy for policy
        # For WDL, we use cross-entropy for value too
        self.value_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, policy_logits: torch.Tensor, value_output: torch.Tensor,
                policy_targets: torch.Tensor, value_targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        AlphaZero loss adapted for WDL:
        - Policy: Cross-entropy between MCTS visit distribution and policy output
        - Value: Cross-entropy between game outcome (WDL) and value output
        - Weight decay handled by AdamW optimizer
        
        Args:
            policy_logits: [batch, policy_size] raw policy logits
            value_output: [batch, 3] WDL logits
            policy_targets: [batch, policy_size] MCTS visit count distribution (soft)
            value_targets: [batch, 3] WDL target distribution
        """
        # Policy loss: Cross-entropy (preserving soft targets)
        # AlphaZero: -π^T log p, then mean over batch
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(policy_targets * policy_log_probs, dim=-1).mean()  # Mean over batch
        
        # Value loss: Cross-entropy for WDL classification
        # Convert soft WDL targets to hard labels if needed
        if value_targets.dim() > 1 and value_targets.shape[1] > 1:
            # If soft targets, convert to hard labels
            value_hard_targets = torch.argmax(value_targets, dim=-1)
        else:
            # Already hard labels
            value_hard_targets = value_targets.long()
        
        value_loss = self.value_loss_fn(value_output, value_hard_targets)  # Already averaged
        
        # Combined loss (weight decay handled by AdamW)
        total_loss = self.value_weight * value_loss + self.policy_weight * policy_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(), 
            'value_loss': value_loss.item()
        }
        
        return total_loss, loss_dict


def simple_train(config: SimpleTrainingConfig):
    """Simple training function"""
    
    # Setup device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {config.model_path}")
    model = load_leela_model(config.model_path, config.config_path, device)
    model.train()
    
    # Load dataset
    logger.info(f"Loading training data from {config.training_data_path}")
    dataset = SimpleDataset(config.training_data_path)
    
    # Train/validation split
    if config.validation_split > 0:
        val_size = int(len(dataset) * config.validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        logger.info(f"Split: {train_size} training, {val_size} validation samples")
    else:
        train_dataset = dataset
        val_dataset = None
        logger.info(f"Using all {len(dataset)} samples for training")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
    
    # Setup training
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    loss_fn = CombinedLoss(
        policy_weight=config.policy_loss_weight,
        value_weight=config.value_loss_weight
    )
    scaler = torch.amp.GradScaler('cuda') if config.use_mixed_precision else None
    
    logger.info(f"Training for {config.epochs} epochs")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Batches per epoch: {len(train_loader)}")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, (positions, policies, values) in enumerate(train_loader):
            positions = positions.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=config.use_mixed_precision):
                policy_logits, value_output = model(positions)
                loss, loss_dict = loss_fn(policy_logits, value_output, policies, values)
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                if config.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
            
            train_losses.append(loss_dict['total_loss'])
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{config.epochs}, "
                           f"Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss_dict['total_loss']:.4f}")
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for positions, policies, values in val_loader:
                    positions = positions.to(device, non_blocking=True)
                    policies = policies.to(device, non_blocking=True)
                    values = values.to(device, non_blocking=True)
                    
                    with torch.amp.autocast('cuda', enabled=config.use_mixed_precision):
                        policy_logits, value_output = model(positions)
                        _, loss_dict = loss_fn(policy_logits, value_output, policies, values)
                    
                    val_losses.append(loss_dict['total_loss'])
            
            avg_val_loss = np.mean(val_losses)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{config.epochs} completed in {epoch_time:.1f}s")
        logger.info(f"  Train loss: {avg_train_loss:.4f}")
        if val_loader:
            logger.info(f"  Val loss: {avg_val_loss:.4f}")
    
    # Save trained model
    logger.info(f"Saving trained model to {config.output_model_path}")
    
    # Load original config for saving
    import json
    with open(config.config_path, 'r') as f:
        model_config = json.load(f)
    
    save_leela_model(
        model,
        Path(config.output_model_path).with_suffix(''),  # Remove .safetensors extension
        config=model_config,
        metadata={
            'trained_epochs': config.epochs,
            'final_train_loss': avg_train_loss,
            'final_val_loss': avg_val_loss,
            'training_samples': len(dataset)
        }
    )
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    logger.info(f"Model saved to {config.output_model_path}")


def main():
    """Main function"""
    
    config = SimpleTrainingConfig(
        # Data
        training_data_path="pimcts_training_data",
        validation_split=0.1,
        
        # Model (matching your selfplay script)
        model_path="eval_small.safetensors",
        config_path="eval_small.json", 
        output_model_path="eval_small_trained.safetensors",
        
        # Training
        batch_size=32,
        learning_rate=3e-4,
        epochs=3,
        
        # Hardware
        device="auto",
        num_workers=4
    )
    
    try:
        simple_train(config)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()