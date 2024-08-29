# simple diffusion transformer

## overview

1. patchify
    image is split into patches
2. position embedding
    learnable position embedding is added to the patches
3. transformer
    patch tokens are passed through transformer encoder
4. decoder
    reconstructs image from the next token patch tokens
5. diffusion
    noise is added to the image and model learns to denoise it at each step

## files

- train.py: contains training loop for the DiT model
- model.py: implements DiT (Diffusion Transformer) model
- transformer.py: defines TransformerBlock, SelfAttention, and LayerNorm
- diffusion.py: defines diffusion process for the model
