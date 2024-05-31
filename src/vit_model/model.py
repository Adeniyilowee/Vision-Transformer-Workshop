import torch
from torch import nn
import torch.nn.functional as F

class Patch_Embeddings(nn.Module):
    """
    A module for creating patch embeddings from input images in a Vision Transformer.

    Args:
        in_channels (int): Number of input channels of the image. Default is 3 (for RGB images).
        patch_size (int): Size of the patches to be extracted from the image. Default is 16.
        embedding_dim (int): Dimension of the embedding space. Default is 768.
        
    Attributes:
        patch_size (int): Stores the size of the patches.
        patcher (nn.Conv2d): Convolutional layer that extracts patches from the input image.
        flatten (nn.Flatten): Layer that flattens the patches.
        
    Methods:
        forward(x):
            Forward pass that converts an input image into patch embeddings.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, num_patches, embedding_dim).
                
    """
    def __init__(self, in_channels:int=3, patch_size:int=16, embedding_dim:int=768):
        super().__init__()

        self.patch_size = patch_size

        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)

    def forward(self, x):
        """
        Forward pass that converts an input image into patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, embedding_dim).
            
        """

        image_resolution = x.shape[-1]

        patched_img = self.patcher(x) 
        flattened_img = self.flatten(patched_img)
        
        return flattened_img.permute(0, 2, 1)

    
class Transformer_Encoder_Block(nn.Module):
    """
    Transformer Encoder Block for Vision Transformer.

    This block consists of a multi-head self-attention (MSA) mechanism followed
    by a multi-layer perceptron (MLP) block with residual connections.

    Args:
        embedding_dim (int): Dimension of the input embeddings. Default is 768.
        num_heads (int): Number of attention heads. Default is 12.
        mlp_size (int): Size of the MLP hidden layer. Default is 3072.
        mlp_dropout (float): Dropout rate for the MLP block. Default is 0.1.
        attn_dropout (float): Dropout rate for the attention mechanism. Default is 0.

    Attributes:
        msa_block (nn.Module): Multi-head self-attention block.
        mlp_block (nn.Module): Multi-layer perceptron block.

    Methods:
        forward(x): Passes the input tensor through the MSA and MLP blocks with residual connections.
        
    """
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from table 1, 768 for ViT-Base
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super().__init__()

        # Create MSA block (equation 2)
        self.msa_block = MSAttention_Block(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # Create MLP block (equation 3)
        self.mlp_block = MLP_Block(embedding_dim=embedding_dim, 
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)
    
    def forward(self, x):
        """
        Forward pass of the Transformer Encoder Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
            
        """
        
        x = self.msa_block(x) + x # residual/skip connection for equation 2
        x = self.mlp_block(x) + x # residual/skip connection for equation 3
        return x 


class Head(nn.Module):
    """
    Attention Head mechanism used in Vision Transformer models.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        head_size (int): The dimension of the attention heads.
        dropout (float): Dropout rate applied to the attention scores.

    Attributes:
        key (nn.Linear): Linear layer to project the input to key vectors.
        query (nn.Linear): Linear layer to project the input to query vectors.
        value (nn.Linear): Linear layer to project the input to value vectors.
        dropout (nn.Dropout): Dropout layer applied to the attention scores.

    Methods:
        forward(query, key, value):
            Computes the attention output given query, key, and value tensors.
        
    """
    def __init__(self, embed_dim, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Computes the attention output given query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: The output of the attention mechanism of shape (batch_size, seq_len, head_size).
            
        """
        B,T,C = key.shape
        
        key = self.key(key) # (B,T,hs)
        query = self.query(query) # (B,T,hs)
        
        # compute attention scores ("affinities")
        matmul_qk = query @ key.transpose(-2,-1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        scores = matmul_qk * key.size(-1)**-0.5 # Scale Factor

        # scores = scores.masked_fill(torch.tril(torch.ones(T,T)) == 0, float('-inf'))# (B, T, T) # this can be commented out for a bi-directional effect
        
        p_attn = F.softmax(scores, dim=-1) # (B, T, T)
        
        p_attn = self.dropout(p_attn)
        
        # perform the weighted aggregation of the values
        value = self.value(value) # (B,T,hs)
        attention = p_attn @ value # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        return attention
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        embed_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float): Dropout rate applied to the attention scores.

    Attributes:
        heads (nn.ModuleList): List of attention heads, each an instance of the `Head` class.
        proj (nn.Linear): Linear layer to project the concatenated outputs of the attention heads.
        dropout (nn.Dropout): Dropout layer applied to the projected attention output.

    Methods:
        forward(query, key, value):
            Computes the multi-head attention output given query, key, and value tensors.
            
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        
        head_size = embed_dim // num_heads
        
        self.heads = nn.ModuleList([Head(embed_dim, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Computes the multi-head attention output given query, key, and value tensors.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, embed_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, embed_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: The output of the multi-head attention mechanism of shape (batch_size, seq_len, embed_dim).
            
        """
        attention = torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        attention = self.dropout(self.proj(attention))
        
        return attention


class MSAttention_Block(nn.Module):
    """
    Multi-Head Self-Attention block.

    Args:
        embedding_dim (int): The dimension of the input embeddings. Default is 768.
        num_heads (int): The number of attention heads. Default is 12.
        attn_dropout (float): Dropout rate applied to the attention scores. Default is 0.

    Attributes:
        layer_norm (nn.LayerNorm): Layer normalization applied to the input.
        multihead_attn (MultiHeadAttention): Multi-head attention layer.

    Methods:
        forward(x):
            Applies layer normalization and multi-head attention to the input tensor.
            
    """
    def __init__(self, embedding_dim:int=768, num_heads:int=12, attn_dropout:float=0):
        super().__init__()
        
        # Layer Norm (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multihead attention (MSA) layer
        self.multihead_attn = MultiHeadAttention(embed_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 dropout=attn_dropout)
        
    def forward(self, x):
        """
        Applies layer normalization and multi-head attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: The output of the attention mechanism of shape (batch_size, seq_len, embedding_dim).
            
        """
        x = self.layer_norm(x)
        
        attention = self.multihead_attn(query=x,
                                        key=x,
                                        value=x)
        
        return attention


class MLP_Block(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block used in Vision Transformer models.

    Args:
        embedding_dim (int): The dimension of the input embeddings. Default is 768.
        mlp_size (int): The size of the hidden layer in the MLP. Default is 3072.
        dropout (float): Dropout rate applied after each linear layer. Default is 0.1.

    Attributes:
        layer_norm (nn.LayerNorm): Layer normalization applied to the input.
        mlp (nn.Sequential): A sequential container of linear layers, activation functions, and dropout layers.

    Methods:
        forward(x):
            Applies layer normalization followed by the MLP to the input tensor.
            
    """
    def __init__(self, embedding_dim:int=768, mlp_size:int=3072, dropout:float=0.1):
        super().__init__()

        # Create the norm layer (LN) 
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the MLP
        self.mlp = nn.Sequential(nn.Linear(in_features=embedding_dim, out_features=mlp_size),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(in_features=mlp_size,
                                           out_features=embedding_dim),
                                 nn.Dropout(p=dropout))

    def forward(self, x):
        """
        Applies layer normalization followed by the MLP to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            torch.Tensor: The output of the MLP block of shape (batch_size, seq_len, embedding_dim).
            
        """        
        return self.mlp(self.layer_norm(x))


class Vision_Transformer(nn.Module):
    """
    Vision Transformer architecture for image classification.
    
    Args:
        img_size (int): Size of the input image (default: 224).
        in_channels (int): Number of input channels of the image (default: 3).
        patch_size (int): Size of the image patches (default: 16).
        num_transformer_layers (int): Number of transformer layers (default: 12).
        embedding_dim (int): Dimensionality of the token embeddings (default: 768).
        mlp_size (int): Size of the feedforward layer in the transformer encoder block (default: 3072).
        num_heads (int): Number of attention heads (default: 12).
        attn_dropout (float): Dropout probability of the attention layers (default: 0).
        mlp_dropout (float): Dropout probability of the MLP layers (default: 0.1).
        embedding_dropout (float): Dropout probability of the token embeddings (default: 0.1).
        num_classes (int): Number of output classes (default: 1000).
    
    Attributes:
        num_patches (int): Number of image patches.
        class_embedding (nn.Parameter): Learnable class token embedding.
        position_embedding (nn.Parameter): Positional embeddings for patches.
        embedding_dropout (nn.Dropout): Dropout layer for token embeddings.
        patch_embedding (Patch_Embeddings): Patch embedding layer.
        transformer_encoder (nn.Sequential): Sequential transformer encoder blocks.
        classifier (nn.Sequential): Sequential classifier layers.
    
    Methods:
        forward(x):
            Applies class embeddings, position embeddings, embedding dropout followed by transformer encoding to the input tensor.
            
    """
    def __init__(self, img_size:int=224, in_channels:int=3, patch_size:int=16, num_transformer_layers:int=12,
                 embedding_dim:int=768, mlp_size:int=3072, num_heads:int=12, attn_dropout:float=0, mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1, num_classes:int=1000):
        super().__init__()

        assert img_size % patch_size == 0,  f"Image size must be divisible by patch size, image: {img_size}, patch size: {patch_size}"
        self.num_patches = (img_size * img_size) // patch_size**2
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True) 
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim))
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = Patch_Embeddings(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[Transformer_Encoder_Block(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim), 
                                        nn.Linear(in_features=embedding_dim, 
                                                  out_features=num_classes),
                                        nn.ReLU())
        
    def forward(self, x):
        """
        Forward pass of the Vision Transformer model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output tensor representing class probabilities.
            
        """
        batch_size = x.shape[0]
        
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim=1) # (batch_size, number_of_patches, embedding_dim)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0])

        return x 