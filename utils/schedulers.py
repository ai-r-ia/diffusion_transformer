# Task 1: Noise scheduler
import torch
import numpy as np
import math

# I'll be using the cosine noise scheduler for this project due to it's promising results (https://arxiv.org/pdf/2102.09672) and further based on this discussion thread: https://ai.stackexchange.com/questions/41068/about-cosine-noise-schedule-in-diffusion-model, but I'll still implement both for choices

import torch
import math

class NoiseScheduler:
    def __init__(self, noise_steps: int = 1000, device: str = 'cpu', s: float = 0.008, beta_min: float = 1e-4, beta_max: float = 0.02, schedule_type: str = 'cosine'):
        self.noise_steps = noise_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.s = s
        self.device = device 

        if schedule_type == 'cosine':
            self._initialize_cosine_schedule()
        elif schedule_type == 'linear':
            self._initialize_linear_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bars = self.alpha_bars.to(self.device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(self.device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(self.device)
        
    def _initialize_cosine_schedule(self):
        T = self.noise_steps
        timesteps = torch.arange(T + 1, dtype=torch.float32)        
        scaled_t = timesteps / T
        term = ((scaled_t + self.s) / (1 + self.s)) * (math.pi / 2)
        f_t = torch.cos(term).pow(2)
        
        alpha_bars_raw = f_t / f_t[0]                
        alpha_bars_raw[1:] = torch.clamp(alpha_bars_raw[1:], min=1e-5, max=0.9999) #keeping alpha_0 = 1 for t = 0 (no noise)
        
        alpha_bar_t = alpha_bars_raw[1:]       
        alpha_bar_t_minus_1 = alpha_bars_raw[:-1] 
        
        self.alphas = alpha_bar_t / alpha_bar_t_minus_1
        
        self.betas = 1.0 - self.alphas
        self.betas = torch.clamp(self.betas, min=self.beta_min, max=self.beta_max)
        
        self.alphas = 1.0 - self.betas 
        self.alpha_bars = self.alphas.cumprod(dim=0) 
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def _initialize_linear_schedule(self):
        T = self.noise_steps
        
        self.betas = torch.linspace(self.beta_min, self.beta_max, self.noise_steps)
        self.alphas = 1.0 - self.betas
        
        self.alpha_bars = self.alphas.cumprod(dim=0) 
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        
    
    # forward sampling
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        
        if noise is None:
            noise = torch.randn_like(x_0)
        
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.index_select(0, t)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.index_select(0, t)
        
        B, C, H, W = x_0.shape
        coeff_shape = (B, 1, 1, 1)
        sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(coeff_shape)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(coeff_shape)
        
        x_t = (sqrt_alpha_bar_t * x_0) + (sqrt_one_minus_alpha_bar_t * noise)
        
        return x_t, noise

    # reverse sampling
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise) -> torch.Tensor:
        B, C, H, W = x_t.shape
        coeff_shape = (B, 1, 1, 1)
        
        beta_t = self.betas.index_select(0, t).view(coeff_shape)
        alpha_t = self.alphas.index_select(0, t).view(coeff_shape)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars.index_select(0, t).view(coeff_shape)
        
        reverse_mean = (1.0/torch.sqrt(alpha_t)) *(x_t  - ((beta_t/sqrt_one_minus_alpha_bar_t)* predicted_noise))
        
        reverse_variance = beta_t 
        
        if t[0].item() == 0:            
            x_t_minus_one = reverse_mean
        else:
            noise = torch.randn_like(x_t, device=x_t.device)            
            x_t_minus_one = reverse_mean + torch.sqrt(reverse_variance) * noise
            
        return x_t_minus_one