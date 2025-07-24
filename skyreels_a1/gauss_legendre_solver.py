import torch

def gauss_legendre_2s_step(model_fn, x, sigma_t, sigma_s, extra_args=None):
    """
    Fast one-pass Gauss-Legendre 2-stage Runge-Kutta sampler (RES4LYF-style).
    
    Args:
        model_fn: A function (x, sigma, **extra_args) -> prediction (noise or velocity)
        x: latent tensor at sigma_t
        sigma_t: starting noise level
        sigma_s: ending noise level
        extra_args: optional dict for model_fn

    Returns:
        x_next: updated latent after one solver step
    """
    if extra_args is None:
        extra_args = {}

    device = x.device
    dtype = x.dtype
    dt = sigma_s - sigma_t

    sqrt_3 = torch.tensor(3.0, device=device, dtype=dtype).sqrt()

    # Butcher tableau coefficients
    c = torch.tensor([0.5 - sqrt_3 / 6, 0.5 + sqrt_3 / 6], device=device, dtype=dtype)
    A = torch.tensor([
        [0.25, 0.25 - sqrt_3 / 6],
        [0.25 + sqrt_3 / 6, 0.25]
    ], device=device, dtype=dtype)
    b = torch.tensor([0.5, 0.5], device=device, dtype=dtype)

    # Stage evaluation (no iteration)
    k = []
    for i in range(2):
        x_i = x.clone()
        for j in range(2):
            x_i = x_i + dt * A[i, j] * (k[j] if j < len(k) else 0)
        sigma_i = sigma_t + c[i] * dt
        k_i = model_fn(x_i, sigma_i, **extra_args)
        k.append(k_i)

    # Final update
    x_next = x + dt * sum(b[i] * k[i] for i in range(2))
    return x_next
