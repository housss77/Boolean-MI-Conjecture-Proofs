import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Use float64 for strict numerical stability.
torch.set_default_dtype(torch.float64)
EPS = 1e-10

# --- 1. Fundamental Math Functions ---

def H2_torch(x):
    """Binary Entropy."""
    x = torch.clamp(x, EPS, 1 - EPS)
    return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

def J_torch(x):
    """Derivative of Binary Entropy: J(x) = log2((1-x)/x)."""
    x = torch.clamp(x, EPS, 1 - EPS)
    return torch.log2((1 - x) / x)

def L_torch(u):
    """L(u) = 2*H2(u) / (1-2u)"""
    u = torch.clamp(u, EPS, 0.5 - EPS)
    return 2 * H2_torch(u) / (1 - 2 * u)

def Linv_solver(y):
    """Numerical solver for L^-1(y). Gradients are not needed for plotting."""
    with torch.no_grad():
        lo = torch.full_like(y, EPS)
        hi = torch.full_like(y, 0.5 - EPS)
        for _ in range(60):
            mid = (lo + hi) / 2
            mask = L_torch(mid) < y
            lo = torch.where(mask, mid, lo)
            hi = torch.where(mask, hi, mid)
        return (lo + hi) / 2

# --- 2. The Target Functions ---

def kappa_torch(u, w):
    """Evaluates kappa(u, w)."""
    diff = u - w
    adiff = torch.abs(diff)
    term1 = diff * (J_torch(w) - J_torch(u)) / 2.0

    # Guard against 0/0 division on the diagonal
    denom = torch.where(adiff < EPS, torch.ones_like(adiff)*EPS, adiff)
    y_arg = (H2_torch(u) + H2_torch(w)) / denom

    inv_L_val = Linv_solver(y_arg)
    term2 = adiff * J_torch(inv_L_val)

    return term1 - term2

def kappa_1_minus_u_torch(u, w):
    """Evaluates kappa(1-u, w). Simplified to remove absolute values since 1-u-w > 0."""
    diff = 1.0 - u - w
    term1 = diff * (J_torch(w) + J_torch(u)) / 2.0

    y_arg = (H2_torch(u) + H2_torch(w)) / diff
    inv_L_val = Linv_solver(y_arg)
    term2 = diff * J_torch(inv_L_val)

    return term1 - term2

def g_func(u, w):
    """g(u, w) = kappa(u, w) - kappa(1-u, w)"""
    return kappa_torch(u, w) - kappa_1_minus_u_torch(u, w)

# --- 3. Execution and 3D Plotting ---

if __name__ == "__main__":
    print("Generating 3D Surface Plot for the Kappa Asymmetry Conjecture...")
    
    # 1. Generate a high-resolution grid over (0, 0.5)^2
    res = 150
    u_vals = np.linspace(0.001, 0.499, res)
    w_vals = np.linspace(0.001, 0.499, res)
    U, W = np.meshgrid(u_vals, w_vals)

    U_t = torch.tensor(U)
    W_t = torch.tensor(W)

    # 2. Compute g(u,w) across the entire grid
    G_t = g_func(U_t, W_t)
    
    # Optional: Mask out the exact microscopic diagonal where the numerical limit approaches 0/0
    # to keep the plot perfectly smooth.
    mask = torch.abs(U_t - W_t) < 1e-4
    G_t[mask] = np.nan
    G = G_t.numpy()

    # Calculate the absolute maximum value to verify it never crosses 0
    max_val = np.nanmax(G)
    print(f"Maximum value of g(u,w) found on grid: {max_val:.8f}")
    
    if max_val <= 1e-10:
        print(">> VERDICT: Empirical evidence strongly supports g(u,w) <= 0 globally.")

    # 3. Render the 3D Surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with a coolwarm color map (reds for high, blues for low)
    surf = ax.plot_surface(U, W, G, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.9)
    
    # Add a flat transparent red plane at Z=0 to represent the rigorous upper bound
    zero_plane = np.zeros_like(G)
    ax.plot_surface(U, W, zero_plane, color='red', alpha=0.15)

    ax.set_title("Surface of g(u,w) = $\kappa(u,w) - \kappa(1-u,w)$ \n (Red Plane represents Z=0 Bound)", fontsize=14)
    ax.set_xlabel('u', fontsize=12)
    ax.set_ylabel('w', fontsize=12)
    ax.set_zlabel('g(u,w)', fontsize=12)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=135)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='g(u,w) value')
    
    plt.tight_layout()
    plt.show()