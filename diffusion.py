import torch

class Diffusion:
    """
    diffusion process
    """
    def __init__(self, config):
        self.beta_start = config.beta_start # start of the beta schedule
        self.beta_end = config.beta_end # end of the beta schedule  
        self.timesteps = config.timesteps # number of timesteps
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps) # beta schedule 
        self.alphas = 1. - self.betas # alpha schedule
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) # cumulative product of alpha schedule
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def diffuse(self, x0, t):
        """
        diffuse the input x0 at time t
        """
        noise = torch.randn_like(x0)
        xt = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0 + \
             self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
        return xt, noise

    def reverse_diffusion_step(self, xt, noise_pred, t):
        """
        reverse the diffusion process at time t
        """
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / alpha_t)
        
        mean = sqrt_recip_alphas_t * (xt - beta_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
        
        if t > 0:
            noise = torch.randn_like(xt)
        else:
            noise = torch.zeros_like(xt)
            
        return mean + torch.sqrt(beta_t) * noise
