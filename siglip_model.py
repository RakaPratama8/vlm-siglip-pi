from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size = 768,
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        num_channels = 3,
        image_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e-6,
        attention_droupout = 0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_droupout = attention_droupout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(nn.Module):
    def __init__(
        self,
        config=SiglipVisionConfig
    ):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(
            num_embeddings=self.num_positions,
            embedding_dim=self.embed_dim,
        )
        
        self.register_buffer(
            name="position_ids",
            tensor=torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_size, Channels, Height, Width]
        
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_size, Embed_dim, Num_Patches_H, Num_Patches_W] -> [Batch_size, Embed_dim, Num_Patches]
        # Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_size, Embed_dim, Num_Patches] -> [Batch_size, Num_Patches, Embed_dim]
        embeddings = embeddings.transpose(1, 2)
        
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_size, Num_Patches, Embed_dim]
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(
        self,
        config=SiglipVisionConfig
    ):
        super().__init__()
        
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_size, Num_Patches, Embed_dim] -> [Batch_size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # [Batch_size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_size, Num_Patches, Intermediate_Size] -> [Batch_size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)
        # [Batch_size, Num_Patches, Embed_Dim]
        return hidden_states

class SiglipAttention(nn.Module):
    def __init__ (
        self,
        config=SiglipVisionConfig
    ):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # 1/sqrt(self.head_dim)
        self.dropout = config.attention_droupout
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # hidden states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        
        # query states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)

class SiglipEncoderLayer(nn.Module):
    def __init__ (
        self,
        config=SiglipVisionConfig
    ):
        super().__init__()
        
        self.embed_dim = config.hidden
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states=torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states
        
class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config=SiglipVisionConfig
    ):
        super().__init__()
        
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoderLayer(config)
        self.post_layernorms = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        
        hidden_states = self.embeddings(pixel_values)
        
        last_hidden_state = self.encoder(input_embeds=hidden_states)
        
        last_hidden_state = self.post_layernorms(last_hidden_state)
        
        return last_hidden_state

class SiglipVisionModel(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig
    ):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values)
