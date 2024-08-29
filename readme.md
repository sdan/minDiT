# simple diffusion transformer
implements https://arxiv.org/pdf/2212.09748 in a simple, clean, and minimal way. uses a adaLN-Zero variant of the transformer block in the DiT. useful for practice not for implementation.

<img width="537" alt="Screenshot 2024-08-28 at 7 19 52 PM" src="https://github.com/user-attachments/assets/5295f6d5-4cf3-4480-94d7-355055860535">

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
