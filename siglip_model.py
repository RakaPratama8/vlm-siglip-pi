from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:
    """
    Class SiglipVisionConfig digunakan untuk mengonfigurasi parameter-parameter dari model Vision Transformer 
    yang digunakan dalam sistem Siglip. Class ini menyediakan berbagai atribut yang dapat disesuaikan 
    untuk mengatur arsitektur dan performa model.
    
    Attributes:
        hidden_size (int): Ukuran dari vektor embeddings yang dihasilkan oleh vision transformer. 
        intermediate_size (int): Ukuran dari linear layer pada feed-forward network. 
        num_hidden_layers (int): Jumlah hidden layers pada transformer.
        num_attention_head (int): Jumlah attention heads pada Multi-Head Attention (MHA) 
            di setiap layer.
        num_channels (int): Jumlah channels pada input gambar. Contoh: RGB memiliki 3 channels. 
        image_size (int): Resolusi gambar input dalam piksel (panjang/lebar). Model ini mendukung 
            gambar dengan ukuran 224x224, 448x448, dan 896x896 piksel.
        patch_size (int): Ukuran patch gambar dalam piksel (panjang/lebar).
        layer_norm_eps (float): Nilai epsilon untuk stabilitas pada layer normalization. 
        attention_droupout (float): Tingkat dropout yang diterapkan pada mekanisme attention. 
        num_image_tokens (int, optional): Jumlah image tokens yang dihasilkan dari patch. 
            Nilai ini dapat dihitung berdasarkan ukuran gambar dan ukuran patch jika tidak diberikan.
        **kwargs: Parameter tambahan yang dapat digunakan untuk konfigurasi lebih lanjut.
    """
    
    def __init__(
        self,
        hidden_size = 768,
        intermediate_size = 3072,
        num_hidden_layers = 12,
        num_attention_head = 12,
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
        self.num_attention_head = num_attention_head
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_droupout = attention_droupout
        self.num_image_tokens = num_image_tokens
        
class SiglipVisionModel(nn.Module):
    """
    Kelas `SiglipVisionModel` adalah turunan dari `nn.Module` yang digunakan untuk membangun model penglihatan (vision model) menggunakan PyTorch.
    
    ## Attributes:
        config (SiglipVisionConfig): Objek konfigurasi yang berisi parameter untuk vision model.
        vision_model (SiglipVisionTransformer): Model vision transformer yang diinisialisasi dengan konfigurasi yang diberikan.
        
    ## Methods:
        forward(pixel_values: Any) -> Tuple:
            Memproses nilai pixel input melalui model vision transformer dan mengembalikan output.
            
    ## Args:
        config (SiglipVisionConfig): Objek konfigurasi yang digunakan untuk menginisialisasi model vision transformer.
    """
    def __init__(
        self,
        config: SiglipVisionConfig
    ):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values)