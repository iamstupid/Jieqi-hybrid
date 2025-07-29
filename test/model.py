import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ActivationFunction:
    """Activation function constants matching the C++ enum"""
    RELU = 0
    MISH = 1
    SELU = 2
    SWISH = 3
    RELU_2 = 4
    NONE = 5


class SwishActivation(nn.Module):
    """PyTorch implementation of Swish activation: x * sigmoid(x)"""
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class SquaredReLU(nn.Module):
    """PyTorch implementation of squared ReLU: (ReLU(x))^2"""
    
    def forward(self, x):
        relu_out = F.relu(x)
        return relu_out * relu_out


def get_activation(activation_type: int) -> nn.Module:
    """Factory function to get activation modules"""
    if activation_type == ActivationFunction.RELU:
        return nn.ReLU()
    elif activation_type == ActivationFunction.MISH:
        return nn.Mish()  # Native PyTorch implementation
    elif activation_type == ActivationFunction.SELU:
        return nn.SELU()
    elif activation_type == ActivationFunction.SWISH:
        return SwishActivation()
    elif activation_type == ActivationFunction.RELU_2:
        return SquaredReLU()
    elif activation_type == ActivationFunction.NONE:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


class SmolgenModule(nn.Module):
    """PyTorch implementation of Smolgen (Small General) attention weight generator"""
    
    def __init__(self, 
                 embedding_size: int,
                 num_heads: int,
                 smolgen_hidden_channels: int,
                 smolgen_hidden_sz: int,
                 smolgen_gen_sz: int,
                 activation_type: int = ActivationFunction.MISH):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.smolgen_gen_sz = smolgen_gen_sz
        
        # Compression layer: [batch, 90, embedding_size] -> [batch, 90*hidden_channels]  
        self.compress = nn.Linear(embedding_size, smolgen_hidden_channels, bias=False)
        
        # First dense layer: [batch, 90*hidden_channels] -> [batch, hidden_sz]
        self.dense1 = nn.Linear(90 * smolgen_hidden_channels, smolgen_hidden_sz, bias=True)
        self.ln1 = nn.LayerNorm(smolgen_hidden_sz, eps=1e-3)
        
        # Second dense layer: [batch, hidden_sz] -> [batch, gen_sz * heads]
        self.dense2 = nn.Linear(smolgen_hidden_sz, smolgen_gen_sz * num_heads, bias=True)
        self.ln2 = nn.LayerNorm(smolgen_gen_sz * num_heads, eps=1e-3)
        
        # Weight generation matrix: [gen_sz, 90*90] for each head
        self.weight_gen = nn.Parameter(torch.randn(smolgen_gen_sz, 90 * 90))
        nn.init.xavier_uniform_(self.weight_gen)
        
        self.activation = get_activation(activation_type)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, 90, embedding_size]
        
        Returns:
            Attention weights of shape [batch, num_heads, 90, 90]
        """
        batch_size = x.shape[0]
        
        # Compression: [batch, 90, embedding_size] -> [batch, 90, hidden_channels]
        flow = self.compress(x)
        
        # Reshape to flatten spatial dimension: [batch, 90*hidden_channels]
        flow = flow.reshape(batch_size, -1)
        
        # First dense + activation + layer norm
        flow = self.dense1(flow)
        flow = self.activation(flow)
        flow = self.ln1(flow)
        
        # Second dense + activation + layer norm
        flow = self.dense2(flow)
        flow = self.activation(flow)
        flow = self.ln2(flow)
        
        # Reshape to [batch, num_heads, gen_sz]
        flow = flow.reshape(batch_size, self.num_heads, self.smolgen_gen_sz)
        
        # Generate attention weights using learned weight matrix
        # [batch, heads, gen_sz] @ [gen_sz, 8100] -> [batch, heads, 8100]
        weights = torch.matmul(flow, self.weight_gen)
        
        # Reshape to attention matrix format: [batch, heads, 90, 90]
        weights = weights.reshape(batch_size, self.num_heads, 90, 90)
        
        return weights


class MultiHeadAttention(nn.Module):
    """PyTorch implementation of multi-head attention with optional Smolgen"""
    
    def __init__(self, 
                 embedding_size: int, 
                 num_heads: int,
                 has_smolgen: bool = False,
                 smolgen_config: Optional[dict] = None):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.d_model = embedding_size
        self.depth = self.d_model // num_heads
        self.has_smolgen = has_smolgen
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_size, self.d_model, bias=True)
        self.k_proj = nn.Linear(embedding_size, self.d_model, bias=True)
        self.v_proj = nn.Linear(embedding_size, self.d_model, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_model, embedding_size, bias=True)
        
        # Smolgen module for dynamic attention weights
        if has_smolgen and smolgen_config:
            self.smolgen = SmolgenModule(
                embedding_size=embedding_size,
                num_heads=num_heads,
                **smolgen_config
            )
        
        self.scale = 1.0 / math.sqrt(self.depth)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.depth)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.depth)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.depth)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, depth]
        K = K.transpose(1, 2).transpose(-2, -1)  # [batch, heads, depth, seq_len]
        V = V.transpose(1, 2)  # [batch, heads, seq_len, depth]
        
        # Compute attention scores
        scores = torch.matmul(Q, K) * self.scale
        
        # Add Smolgen weights if available
        if self.has_smolgen:
            smolgen_weights = self.smolgen(x)
            scores = scores + smolgen_weights
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        if self.num_heads > 1:
            out = out.transpose(1, 2).contiguous()
        
        out = out.view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(out)
        
        return output


class FeedForward(nn.Module):
    """PyTorch implementation of feed-forward network"""
    
    def __init__(self, 
                 embedding_size: int,
                 dff_size: int,
                 activation_type: int = ActivationFunction.RELU,
                 alpha: float = 1.0):
        super().__init__()
        
        self.dense1 = nn.Linear(embedding_size, dff_size, bias=True)
        self.activation = get_activation(activation_type)
        self.dense2 = nn.Linear(dff_size, embedding_size, bias=True)
        self.alpha = alpha
    
    def forward(self, x):
        residual = x
        
        flow = self.dense1(x)
        flow = self.activation(flow)
        flow = self.dense2(flow)
        
        if self.alpha != 1.0:
            flow = flow * self.alpha
        
        # Residual connection
        output = flow + residual
        
        return output


class EncoderLayer(nn.Module):
    """PyTorch implementation of transformer encoder layer with optional Smolgen"""
    
    def __init__(self, 
                 embedding_size: int,
                 num_heads: int,
                 dff_size: int,
                 activation_type: int = ActivationFunction.RELU,
                 alpha: float = 1.0,
                 eps: float = 1e-6,
                 has_smolgen: bool = False,
                 smolgen_config: Optional[dict] = None):
        super().__init__()
        
        self.mha = MultiHeadAttention(
            embedding_size, 
            num_heads, 
            has_smolgen=has_smolgen,
            smolgen_config=smolgen_config
        )
        self.ffn = FeedForward(embedding_size, dff_size, activation_type, alpha)
        
        self.ln1 = nn.LayerNorm(embedding_size, eps=eps)
        self.ln2 = nn.LayerNorm(embedding_size, eps=eps)
        
        self.alpha = alpha
    
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_out = self.mha(x)
        
        if self.alpha != 1.0:
            attn_out = attn_out * self.alpha
        
        x = x + attn_out
        x = self.ln1(x)
        
        # Feed-forward with residual connection (handled inside FFN)
        x = self.ffn(x)
        x = self.ln2(x)
        
        return x


class AttentionBody(nn.Module):
    """PyTorch implementation of attention body with dense positional embedding"""
    
    def __init__(self,
                 input_channels: int,
                 embedding_size: int,
                 num_encoder_layers: int,
                 num_heads: int,
                 dff_size: int,
                 activation_type: int = ActivationFunction.RELU,
                 eps: float = 1e-6,
                 has_smolgen: bool = False,
                 smolgen_config: Optional[dict] = None,
                 embedding_dense_size: int = 16):
        super().__init__()
        
        self.embedding_dense_size = embedding_dense_size
        
        # Dense positional embedding preprocessing
        # Processes first 16 channels (positional info) -> embedding_dense_size per square
        self.pos_preprocess = nn.Linear(90 * 16, 90 * embedding_dense_size, bias=True)
        
        # Input embedding: maps from (input_channels + embedding_dense_size) to embedding_size
        effective_input_size = input_channels + embedding_dense_size
        self.input_embedding = nn.Linear(effective_input_size, embedding_size, bias=True)
        self.activation = get_activation(activation_type)
        
        # Input layer normalization (eps=1e-3 for dense embedding)
        self.input_ln = nn.LayerNorm(embedding_size, eps=1e-3)
        
        # Optional multiplicative and additive gating (can be added later)
        self.mult_gate = None
        self.add_gate = None
        
        # Input FFN for dense embedding
        alpha = (2.0 * num_encoder_layers) ** -0.25
        self.input_ffn = FeedForward(
            embedding_size, 
            dff_size, 
            activation_type, 
            alpha
        )
        self.input_ffn_ln = nn.LayerNorm(embedding_size, eps=1e-3)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                embedding_size, 
                num_heads, 
                dff_size, 
                activation_type, 
                alpha, 
                eps,
                has_smolgen=has_smolgen,
                smolgen_config=smolgen_config
            )
            for _ in range(num_encoder_layers)
        ])
    
    def add_gating_layers(self, mult_gate_weights: Optional[torch.Tensor] = None,
                         add_gate_weights: Optional[torch.Tensor] = None):
        """Add optional multiplicative and additive gating layers"""
        if mult_gate_weights is not None:
            self.mult_gate = nn.Parameter(mult_gate_weights)
        if add_gate_weights is not None:
            self.add_gate = nn.Parameter(add_gate_weights)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, seq_len, channel]
        
        Returns:
            Encoded features of shape [batch, seq_len, embedding_size]
        """
        batch_size, tokens, channels = x.shape
        
        # Apply dense positional embedding
        x = self._apply_dense_positional_embedding(x)
        
        # Input embedding and activation
        x = self.input_embedding(x)
        x = self.activation(x)
        
        # Apply input layer normalization
        x = self.input_ln(x)
        
        # Apply gating if present
        if self.mult_gate is not None:
            x = x.view(batch_size, 90, -1)
            x = x * self.mult_gate.unsqueeze(0)
            x = x.view(batch_size, -1)
            
        if self.add_gate is not None:
            x = x.view(batch_size, 90, -1)
            x = x + self.add_gate.unsqueeze(0)
            x = x.view(batch_size, -1)
        
        # Apply input FFN
        x = self.input_ffn(x)
        x = self.input_ffn_ln(x)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def _apply_dense_positional_embedding(self, x):
        """
        Apply dense positional embedding strategy.
        Extracts first 16 channels as positional info and processes them.
        """
        batch_size, seq_len, channels = x.shape
        
        # Extract positional information from first 14 channels
        pos_info = x[:, :, :16]  # [batch, 90, 16]
        
        # Flatten positional info
        pos_info_flat = pos_info.reshape(batch_size, -1)  # [batch, 90*16]
        
        # Process through preprocessing layer
        pos_processed = self.pos_preprocess(pos_info_flat)  # [batch, 90*embedding_dense_size]
        
        # Reshape back to spatial format
        pos_processed = pos_processed.reshape(batch_size, seq_len, self.embedding_dense_size)
        
        # Concatenate with original input
        x_with_pos = torch.cat([x, pos_processed], dim=-1)
        
        return x_with_pos


class PolicyFilter(nn.Module):
    """Policy filtering layer that maps 8100 raw policy logits to valid moves only"""
    
    def __init__(self, policy_index_array: torch.Tensor, num_possible_policies: int = 2550):
        """
        Args:
            policy_index_array: Tensor of shape [8100] where each element is either:
                                - A unique index in [0, num_possible_policies) for valid moves
                                - -1 for invalid moves
            num_possible_policies: Number of possible valid policies (default 2550)
        """
        super().__init__()
        
        self.num_possible_policies = num_possible_policies
        
        # Register policy index array as buffer (not trainable parameter)
        self.register_buffer('policy_index_array', policy_index_array.long())
        
        # Create mask for valid moves (where index != -1)
        valid_mask = (policy_index_array != -1)
        self.register_buffer('valid_mask', valid_mask)
    
    def forward(self, policy_logits_8100):
        """
        Args:
            policy_logits_8100: Raw policy logits of shape [batch, 8100]
        
        Returns:
            filtered_policy: Policy logits of shape [batch, num_possible_policies]
        """
        batch_size = policy_logits_8100.shape[0]
        
        # Initialize output with zeros (or -inf for proper masking)
        output = torch.zeros(
            batch_size, self.num_possible_policies,
            dtype=policy_logits_8100.dtype,
            device=policy_logits_8100.device
        )
        
        # Simple vectorized implementation matching your loop:
        # For each valid position, copy logit to corresponding output position
        valid_indices = self.policy_index_array[self.valid_mask]  # Extract valid target indices
        valid_logits = policy_logits_8100[:, self.valid_mask]     # Extract valid logits
        
        # This is equivalent to your loop but vectorized:
        # output[pol_ind[ind]] = policy_logits[ind] for all valid ind
        output[:, valid_indices] = valid_logits
        
        return output


class AttentionPolicyHead(nn.Module):
    """PyTorch implementation of attention-based policy head with filtering"""
    
    def __init__(self,
                 embedding_size: int,
                 policy_embedding_size: int,
                 policy_d_model: int,
                 num_policy_encoder_layers: int = 0,
                 policy_head_count: int = 8,
                 activation_type: int = ActivationFunction.SELU,
                 policy_index_array: Optional[torch.Tensor] = None,
                 num_possible_policies: int = 2550):
        super().__init__()
        
        # First dense layer
        self.dense1 = nn.Linear(embedding_size, policy_embedding_size, bias=True)
        self.activation = get_activation(activation_type)
        
        # Optional policy encoder layers
        self.policy_encoders = nn.ModuleList([
            EncoderLayer(
                policy_embedding_size,
                policy_head_count,
                policy_embedding_size * 4,  # Typical FFN expansion
                activation_type
            )
            for _ in range(num_policy_encoder_layers)
        ])
        
        # Q, K projections for final attention
        self.q_proj = nn.Linear(policy_embedding_size, policy_d_model, bias=True)
        self.k_proj = nn.Linear(policy_embedding_size, policy_d_model, bias=True)
        
        self.scale = 1.0 / math.sqrt(policy_d_model)
        
        # Policy filtering layer
        self.use_filtering = policy_index_array is not None
        if self.use_filtering:
            self.policy_filter = PolicyFilter(policy_index_array, num_possible_policies)
        
        self.num_possible_policies = num_possible_policies if self.use_filtering else 8100
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, 90, embedding_size]
        
        Returns:
            Policy logits of shape [batch, num_possible_policies]
        """
        batch_size, seq_len, _ = x.shape
        
        # First dense layer
        flow = self.dense1(x)
        flow = self.activation(flow)
        
        # Apply policy encoder layers if any
        for encoder in self.policy_encoders:
            flow = encoder(flow)
        
        # Compute Q and K for attention-based policy
        Q = self.q_proj(flow)  # [batch, 90, policy_d_model]
        K = self.k_proj(flow)  # [batch, 90, policy_d_model]
        
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Flatten to [batch, 90*90] = [batch, 8100]
        policy_logits_8100 = scores.view(batch_size, -1)
        
        # Apply policy filtering if enabled
        if self.use_filtering:
            policy_logits = self.policy_filter(policy_logits_8100)
        else:
            policy_logits = policy_logits_8100
        
        return policy_logits


class ValueHead(nn.Module):
    """PyTorch implementation of value head"""
    
    def __init__(self,
                 embedding_size: int,
                 val_channels: int = 32,
                 is_wdl: bool = True,
                 activation_type: int = ActivationFunction.RELU):
        super().__init__()
        
        self.embedding = nn.Linear(embedding_size, val_channels, bias=True)
        self.activation1 = get_activation(activation_type)
        
        self.dense1 = nn.Linear(val_channels * 90, 128, bias=True)  # 90 = 10*9 board squares
        self.activation2 = get_activation(activation_type)
        
        self.is_wdl = is_wdl
        if is_wdl:
            self.dense2 = nn.Linear(128, 3, bias=True)  # Win/Draw/Loss
        else:
            self.dense2 = nn.Linear(128, 1, bias=True)  # Single value
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, 90, embedding_size]
        
        Returns:
            Value output: [batch, 3] for WDL or [batch, 1] for single value
        """
        batch_size = x.shape[0]
        
        # Embedding layer
        flow = self.embedding(x)
        flow = self.activation1(flow)
        
        # Flatten to [batch, val_channels * 90]
        flow = flow.view(batch_size, -1)
        
        # Dense layers
        flow = self.dense1(flow)
        flow = self.activation2(flow)
        flow = self.dense2(flow)
        
        if self.is_wdl:
            # Apply softmax for WDL
            output = F.softmax(flow, dim=-1)
        else:
            # Apply tanh for single value
            output = torch.tanh(flow)
        
        return output


class LeelaZeroNet(nn.Module):
    """Complete Leela Chess Zero network with attention body and dense positional embedding"""
    
    def __init__(self,
                 input_channels: int = 167,  # kInputPlanes
                 embedding_size: int = 512,
                 dff_size: int = 1024,
                 num_encoder_blocks: int = 8,
                 num_heads: int = 8,
                 policy_embedding_size: int = 256,
                 policy_d_model: int = 128,
                 activation_type: int = ActivationFunction.RELU,
                 is_wdl: bool = True,
                 has_smolgen: bool = False,
                 smolgen_config: Optional[dict] = None,
                 embedding_dense_size: int = 16,
                 policy_index_array: Optional[torch.Tensor] = None,
                 num_policy_encoder_layers = 0,
                 num_possible_policies: int = 2550):
        super().__init__()
        
        # Attention body with dense positional embedding
        self.attention_body = AttentionBody(
            input_channels=input_channels,
            embedding_size=embedding_size,
            num_encoder_layers=num_encoder_blocks,
            num_heads=num_heads,
            dff_size=dff_size, # LCZero stated that large FFN expansion is not very helpful
            activation_type=activation_type,
            has_smolgen=has_smolgen,
            smolgen_config=smolgen_config,
            embedding_dense_size=embedding_dense_size
        )
        
        # Policy head with filtering
        self.policy_head = AttentionPolicyHead(
            embedding_size=embedding_size,
            policy_embedding_size=policy_embedding_size,
            policy_d_model=policy_d_model,
            activation_type=activation_type,
            policy_index_array=policy_index_array,
            num_possible_policies=num_possible_policies,
            num_policy_encoder_layers = num_policy_encoder_layers
        )
        
        # Value head
        self.value_head = ValueHead(
            embedding_size=embedding_size,
            is_wdl=is_wdl,
            activation_type=activation_type
        )
        
        self.num_possible_policies = self.policy_head.num_possible_policies
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, input_channel, 10, 9] (chess board representation)
        
        Returns:
            policy: Policy logits [batch, num_possible_policies] (filtered or 8100)
            value: Value output [batch, 3] for WDL or [batch, 1] for single value
        """
        # Attention body: [batch, input_channels, 10, 9] -> [batch, 90, embedding_size]
        features = self.attention_body(x)
        
        # Policy head: [batch, 90, embedding_size] -> [batch, num_possible_policies]
        policy = self.policy_head(features)
        
        # Value head: [batch, 90, embedding_size] -> [batch, 3] or [batch, 1]
        value = self.value_head(features)
        
        return policy, value
