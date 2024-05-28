import torch
from torch import nn
import torch.nn.functional as F

class Patch_Embeddings(nn.Module):

    def __init__(self,
               in_channels:int=3,
               patch_size:int=16,
               embedding_dim:int=768): # from Table 1 for ViT-Base
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

        image_resolution = x.shape[-1]

        patched_img = self.patcher(x) 
        flattened_img = self.flatten(patched_img)
        
        return flattened_img.permute(0, 2, 1)

 
    
class Transformer_Encoder_Block(nn.Module):
    def __init__(self,
           embedding_dim:int=768, # Hidden size D from table 1, 768 for ViT-Base
           num_heads:int=12, # from table 1
           mlp_size:int=3072, # from table 1
           mlp_dropout:float=0.1, # from table 3
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
        x = self.msa_block(x) + x # residual/skip connection for equation 2
        x = self.mlp_block(x) + x # residual/skip connection for equation 3
        return x 
    

    
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, embed_dim, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        #self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # input of size (batch, latent-space, feature map)
        # output of size (batch, latent-space, head size)
        
        B,T,C = key.shape
        
        key = self.key(key)   # (B,T,hs)
        query = self.query(query) # (B,T,hs)
        
        # compute attention scores ("affinities")
        weight = query @ key.transpose(-2,-1) # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        weight = weight * key.shape[-1]**-0.5 # Scale Factor

        weight = weight.masked_fill(torch.tril(torch.ones(T,T)) == 0, float('-inf'))# (B, T, T) # this can be commented out for a bi-directional effect
        
        weight = F.softmax(weight, dim=-1) # (B, T, T)
        
        weight = self.dropout(weight)
        
        # perform the weighted aggregation of the values
        value = self.value(value) # (B,T,hs)
        out = weight @ value # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        
        return out
    
class MultiHeadAttention(nn.Module):
    """ 
    multiple heads of self-attention in parallel 
    
    """

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        
        head_size = embed_dim // num_heads
        
        self.heads = nn.ModuleList([Head(embed_dim, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        output = torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        
        output = self.dropout(self.proj(output))
        
        return output

class MSAttention_Block(nn.Module): 
    """
    Creates a multi-head self-attention block
    
    """
    
    def __init__(self, embedding_dim:int=768, num_heads:int=12, attn_dropout:float=0): # Heads from Table 1 for ViT-Base
        super().__init__()
        
        # Layer Norm (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Multihead attention (MSA) layer
        self.multihead_attn = MultiHeadAttention(embed_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 dropout=attn_dropout)
        '''
        # Multihead attention (MSA) layer -- In built in PyTorch
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout, 
                                                    batch_first=True) # is the batch first? (batch, seq, feature) -> (batch, number_of_patches, embedding_dimension)
        '''
        
    def forward(self, x):
        x = self.layer_norm(x)
        
        attn_output = self.multihead_attn(query=x,
                                          key=x,
                                          value=x)
        
        '''    
        attn_output, _ = self.multihead_attn2(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=False)
        '''
        return attn_output


    
    
class MLP_Block(nn.Module):
    def __init__(self, embedding_dim:int=768, mlp_size:int=3072, dropout:float=0.1):
        super().__init__()

        # Create the norm layer (LN) 
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the MLP
        self.mlp = nn.Sequential(nn.Linear(in_features=embedding_dim, out_features=mlp_size),
                                 nn.GELU(),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(in_features=mlp_size,
                                           out_features=embedding_dim), # squeezed back to the embedding dimension as seen in the next cell (shape)
                                 nn.GELU(),
                                 nn.Dropout(p=dropout))                 # drop out or activation fn does not chage dimension

    def forward(self, x):
        
        return self.mlp(self.layer_norm(x))



class Vision_Transformer(nn.Module): 
    def __init__(self,
               img_size:int=224, # Table 3 from the ViT paper
               in_channels:int=3,
               patch_size:int=16, 
               num_transformer_layers:int=12, # Table 1 for "Layers" for ViT-Base
               embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
               mlp_size:int=3072, # Table 1
               num_heads:int=12, # Table 1
               attn_dropout:float=0,
               mlp_dropout:float=0.1,
               embedding_dropout:float=0.1, # Dropout for patch and position embeddings
               num_classes:int=1000): # number of classes in our classification problem
        super().__init__()

        # Make an assertion that the image size is compatible with the patch size
        assert img_size % patch_size == 0,  f"Image size must be divisible by patch size, image: {img_size}, patch size: {patch_size}"

        # Calculate the number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # Create learnable position embedding 
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim))

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create patch embedding layer
        self.patch_embedding = Patch_Embeddings(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # Create the Transformer Encoder block ... create stacked
        self.transformer_encoder = nn.Sequential(*[Transformer_Encoder_Block(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # Create classifier head
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim), 
                                        nn.Linear(in_features=embedding_dim, 
                                                  out_features=num_classes),
                                        nn.ReLU())
        
        
  
    def forward(self, x):
        # Get the batch size
        batch_size = x.shape[0]

        # Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimensions

        # Create the patch embedding (equation 1)
        x = self.patch_embedding(x)

        # Concat class token embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1) # (batch_size, number_of_patches, embedding_dim)

        # Add position embedding to class token and patch embedding
        x = self.position_embedding + x

        # Apply dropout to patch embedding ("directly after adding positional- to patch embeddings")
        x = self.embedding_dropout(x)

        # Pass position and patch embedding to Transformer Encoder (equation 2 & 3)
        x = self.transformer_encoder(x)

        # Put 0th index logit through classifier (equation 4)
        x = self.classifier(x[:, 0])

        return x 