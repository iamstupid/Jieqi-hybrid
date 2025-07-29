import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from safetensors.torch import save_file, load_file
import logging

from model import LeelaZeroNet, ActivationFunction

logger = logging.getLogger(__name__)


class LeelaZeroNetConfig:
    """Configuration class for LeelaZeroNet with validation and defaults"""
    
    DEFAULT_CONFIG = {
        'input_channels': 167,
        'embedding_size': 512,
        'dff_size': 1024,
        'num_encoder_blocks': 8,
        'num_heads': 8,
        'policy_embedding_size': 256,
        'policy_d_model': 128,
        'activation_type': 0,  # ActivationFunction.RELU
        'is_wdl': True,
        'has_smolgen': False,
        'smolgen_config': None,
        'embedding_dense_size': 16,
        'num_possible_policies': 2550,
        'policy_index_array': None
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill in default values for configuration"""
        validated_config = cls.DEFAULT_CONFIG.copy()
        validated_config.update(config)
        
        # Validation checks
        assert validated_config['embedding_size'] > 0, "embedding_size must be positive"
        assert validated_config['num_encoder_blocks'] > 0, "num_encoder_blocks must be positive"
        assert validated_config['num_heads'] > 0, "num_heads must be positive"
        assert validated_config['embedding_size'] % validated_config['num_heads'] == 0, \
            "embedding_size must be divisible by num_heads"
        assert validated_config['activation_type'] in range(6), \
            "activation_type must be between 0-5"
        
        # Smolgen validation
        if validated_config['has_smolgen']:
            if validated_config['smolgen_config'] is None:
                # Provide default smolgen config
                validated_config['smolgen_config'] = {
                    'smolgen_hidden_channels': 64,
                    'smolgen_hidden_sz': 128,
                    'smolgen_gen_sz': 32,
                    'activation_type': validated_config['activation_type']
                }
            else:
                required_keys = ['smolgen_hidden_channels', 'smolgen_hidden_sz', 'smolgen_gen_sz']
                for key in required_keys:
                    assert key in validated_config['smolgen_config'], \
                        f"Missing required smolgen_config key: {key}"
        
        return validated_config


class LeelaZeroNetFactory:
    """Factory class for creating, loading, and saving LeelaZeroNet models"""
    
    @staticmethod
    def create_model(config: Union[Dict[str, Any], str, Path]) -> LeelaZeroNet:
        """
        Create a LeelaZeroNet model from configuration
        
        Args:
            config: Either a dict with model parameters, or path to JSON config file
            
        Returns:
            LeelaZeroNet: Initialized model
        """
        if isinstance(config, (str, Path)):
            config = LeelaZeroNetFactory.load_config(config)
        
        validated_config = LeelaZeroNetConfig.validate_config(config)
        
        # Handle policy_index_array if provided as list
        if 'policy_index_array' in validated_config and validated_config['policy_index_array'] is not None:
            if isinstance(validated_config['policy_index_array'], list):
                validated_config['policy_index_array'] = torch.tensor(
                    validated_config['policy_index_array'], dtype=torch.long
                )
        
        logger.info(f"Creating LeelaZeroNet with config: {validated_config}")
        
        return LeelaZeroNet(**validated_config)
    
    @staticmethod
    def save_model(model: LeelaZeroNet, 
                   save_path: Union[str, Path], 
                   config: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights and configuration using safetensors
        
        Args:
            model: The LeelaZeroNet model to save
            save_path: Path to save the model (without extension)
            config: Model configuration dict (optional)
            metadata: Additional metadata to save (optional)
        """
        save_path = Path(save_path)
        
        # Save model weights using safetensors
        weights_path = save_path.with_suffix('.safetensors')
        
        # Prepare metadata for safetensors
        safetensors_metadata = {}
        if metadata:
            # Convert metadata to strings (safetensors requires string values)
            for key, value in metadata.items():
                safetensors_metadata[f"metadata_{key}"] = str(value)
        
        # Add model info to metadata
        safetensors_metadata.update({
            "model_type": "LeelaZeroNet",
            "framework": "pytorch",
            "num_parameters": str(sum(p.numel() for p in model.parameters())),
            "num_trainable_parameters": str(sum(p.numel() for p in model.parameters() if p.requires_grad))
        })
        
        save_file(model.state_dict(), weights_path, metadata=safetensors_metadata)
        logger.info(f"Model weights saved to {weights_path}")
        
        # Save configuration as JSON
        if config is not None:
            config_path = save_path.with_suffix('.json')
            
            # Convert tensors to lists for JSON serialization
            json_config = {}
            for key, value in config.items():
                if isinstance(value, torch.Tensor):
                    json_config[key] = value.tolist()
                else:
                    json_config[key] = value
            
            with open(config_path, 'w') as f:
                json.dump(json_config, f, indent=2)
            logger.info(f"Model config saved to {config_path}")
    
    @staticmethod
    def load_model(model_path: Union[str, Path], 
                   config_path: Optional[Union[str, Path]] = None,
                   device: Optional[torch.device] = None,
                   strict: bool = True) -> LeelaZeroNet:
        """
        Load a LeelaZeroNet model from saved files
        
        Args:
            model_path: Path to the safetensors model file
            config_path: Path to the JSON config file (optional, will try to infer)
            device: Device to load the model on
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            LeelaZeroNet: Loaded model
        """
        model_path = Path(model_path)
        
        # Try to infer config path if not provided
        if config_path is None:
            config_path = model_path.with_suffix('.json')
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            config_path = Path(config_path)
        
        # Load configuration
        config = LeelaZeroNetFactory.load_config(config_path)
        
        # Create model
        model = LeelaZeroNetFactory.create_model(config)
        
        # Load weights
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        state_dict = load_file(model_path, device=str(device))
        model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        
        logger.info(f"Model loaded from {model_path} on device {device}")
        
        return model
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Config loaded from {config_path}")
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        config_path = Path(config_path)
        
        # Convert tensors to lists for JSON serialization
        json_config = {}
        for key, value in config.items():
            if isinstance(value, torch.Tensor):
                json_config[key] = value.tolist()
            else:
                json_config[key] = value
        
        with open(config_path, 'w') as f:
            json.dump(json_config, f, indent=2)
        
        logger.info(f"Config saved to {config_path}")
    
    @staticmethod
    def get_model_info(model_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a saved model without loading it"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load metadata from safetensors file
        from safetensors import safe_open
        
        info = {}
        with safe_open(model_path, framework="pt") as f:
            info['metadata'] = dict(f.metadata()) if f.metadata() else {}
            info['tensor_names'] = f.keys()
            
            # Calculate total parameters
            total_params = 0
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_params += tensor.numel()
            info['total_parameters'] = total_params
        
        # Try to load config if available
        config_path = model_path.with_suffix('.json')
        if config_path.exists():
            info['config'] = LeelaZeroNetFactory.load_config(config_path)
        
        return info


# Convenience functions
def create_leela_model(config: Union[Dict[str, Any], str, Path]) -> LeelaZeroNet:
    """Convenience function to create a model"""
    return LeelaZeroNetFactory.create_model(config)


def save_leela_model(model: LeelaZeroNet, 
                     save_path: Union[str, Path], 
                     config: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to save a model"""
    LeelaZeroNetFactory.save_model(model, save_path, config, metadata)


def load_leela_model(model_path: Union[str, Path], 
                     config_path: Optional[Union[str, Path]] = None,
                     device: Optional[torch.device] = None) -> LeelaZeroNet:
    """Convenience function to load a model"""
    return LeelaZeroNetFactory.load_model(model_path, config_path, device)


# Example usage and configuration templates
EXAMPLE_CONFIGS = {
    "default": {
        "input_channels": 167,
        "embedding_size": 768,
        "dff_size": 1024,
        "num_encoder_blocks": 15,
        "num_heads": 12,
        "policy_embedding_size": 256,
        "policy_d_model": 128,
        "activation_type": ActivationFunction.MISH,
        "is_wdl": True,
        "has_smolgen": True,
        "embedding_dense_size": 32,
        "num_possible_policies": 2550,
        "smolgen_config":{
            'smolgen_hidden_channels': 32,
            'smolgen_hidden_sz': 256,
            'smolgen_gen_sz': 256,
            'activation_type': ActivationFunction.MISH
        }
    },
    "small": {
        "input_channels": 167,
        "embedding_size": 512,
        "dff_size": 768,
        "num_encoder_blocks": 10,
        "num_heads": 8,
        "policy_embedding_size": 256,
        "policy_d_model": 128,
        "activation_type": ActivationFunction.MISH,
        "is_wdl": True,
        "has_smolgen": True,
        "embedding_dense_size": 16,
        "num_possible_policies": 2550,
        "smolgen_config":{
            'smolgen_hidden_channels': 32,
            'smolgen_hidden_sz': 128,
            'smolgen_gen_sz': 128,
            'activation_type': ActivationFunction.MISH
        }
    }
}


if __name__ == "__main__":
    # Example usage
    
    # Create a model from config
    config = EXAMPLE_CONFIGS["small"]
    model = create_leela_model(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Save the model
    save_leela_model(
        model, 
        "my_leela_model", 
        config=config,
        metadata={"version": "1.0", "training_steps": 10000}
    )
    
    # Load the model
    loaded_model = load_leela_model("my_leela_model.safetensors")
    print("Model loaded successfully")
    
    # Get model info without loading
    info = LeelaZeroNetFactory.get_model_info("my_leela_model.safetensors")
    print(f"Model info: {info['total_parameters']} parameters")