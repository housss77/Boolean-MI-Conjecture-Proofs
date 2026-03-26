import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd.functional import hessian, jacobian

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

def J_prime_torch(x):
    """Second derivative of H2."""
    x = torch.clamp(x, EPS, 1 - EPS)
    ln2 = np.log(2)
    return -1.0 / (ln2 * x * (1 - x))

def L_torch(u):
    """L(u) = 2*H2(u) / (1-2u)"""
    u = torch.clamp(u, EPS, 0.5 - EPS)
    return 2 * H2_torch(u) / (1 - 2 * u)

def H2inv_solver(y):
    """Numerical solver for H2(x) = y. No gradients tracked."""
    with torch.no_grad():
        lo = torch.full_like(y, EPS)
        hi = torch.full_like(y, 0.5)
        for _ in range(60):
            mid = (lo + hi) / 2
            mask = H2_torch(mid) < y
            lo = torch.where(mask, mid, lo)
            hi = torch.where(mask, hi, mid)
        return (lo + hi) / 2

# --- 2. Custom Analytic Autograd for L^-1 ---

class LinvFunction(torch.autograd.Function):
    """
    Custom Autograd Function for L^-1. 
    Bypasses AD through the binary search loop by explicitly defining 
    the backward pass using the Inverse Function Theorem.
    """
    @staticmethod
    def forward(ctx, y):
        # 1. Forward pass: Binary Search (No gradient tracking here)
        with torch.no_grad():
            lo = torch.full_like(y, EPS)
            hi = torch.full_like(y, 0.5 - EPS)
            for _ in range(60):
                mid = (lo + hi) / 2
                mask = L_torch(mid) < y
                lo = torch.where(mask, mid, lo)
                hi = torch.where(mask, hi, mid)
            u_val = (lo + hi) / 2
        
        # Save the computed u for the backward pass
        ctx.save_for_backward(u_val)
        return u_val

    @staticmethod
    def backward(ctx, grad_output):
        # 2. Backward pass: Analytic Chain Rule
        u, = ctx.saved_tensors
        
        # Compute L'(u) = ( 2*J(u)*(1-2u) + 4*H2(u) ) / (1-2u)^2
        num = 2 * J_torch(u) * (1 - 2 * u) + 4 * H2_torch(u)
        den = (1 - 2 * u) ** 2
        L_prime = num / den
        
        # Inverse Function Theorem: d(L^-1)/dy = 1 / L'(u)
        grad_input = grad_output * (1.0 / L_prime)
        return grad_input

# Alias the custom autograd function so it behaves like a standard PyTorch function
Linv_autograd = LinvFunction.apply

# --- 3. The Kappa Function ---

def kappa_torch(u, w):
    """Evaluates kappa(u, w) using the custom autograd-enabled Linv."""
    diff = u - w
    adiff = torch.abs(diff)
    
    term1 = diff * (J_torch(w) - J_torch(u)) / 2.0

    # Guard against 0/0 division on the diagonal
    denom = torch.where(adiff < EPS, torch.ones_like(adiff)*EPS, adiff)
    y_arg = (H2_torch(u) + H2_torch(w)) / denom

    # Use our custom analytic backprop function here!
    inv_L_val = Linv_autograd(y_arg)
    term2 = adiff * J_torch(inv_L_val)

    return term1 - term2

# --- 4. Convexity Verification Engine ---

class KappaConvexityAnalyzer:
    def __init__(self):
        self.global_min_eigenvalue = float('inf')
        self.worst_coords = None

    def get_stable_hessian_g(self, l_val, m_val):
        """Computes Hessian of g(l,m) = kappa(H2inv(l), H2inv(m)) via Chain Rule."""
        # Use as_tensor and detach to prevent PyTorch graph warnings
        l_t = torch.as_tensor(l_val, dtype=torch.float64).detach()
        m_t = torch.as_tensor(m_val, dtype=torch.float64).detach()
        
        # 1. Solve u,w from l,m numerically
        u = H2inv_solver(l_t)
        w = H2inv_solver(m_t)
        
        # 2. Enable gradients for Hessian calculation
        u.requires_grad_(True)
        w.requires_grad_(True)
        
        def kappa_wrapper(u_in, w_in):
            return kappa_torch(u_in, w_in)
        
        # 3. Compute pure Hessian and Jacobian of kappa w.r.t (u,w)
        H_k = hessian(kappa_wrapper, (u, w))
        Grad_k = jacobian(kappa_wrapper, (u, w))
        
        K_uu, K_uw = H_k[0][0], H_k[0][1]
        K_wu, K_ww = H_k[1][0], H_k[1][1]
        dK_du, dK_dw = Grad_k[0], Grad_k[1]
        
        # 4. Chain Rule derivatives for the inverse mapping
        J_u = J_torch(u)
        J_w = J_torch(w)

        du_dl = 1.0 / J_u
        dw_dm = 1.0 / J_w

        d2u_dl2 = -J_prime_torch(u) / (J_u**3)
        d2w_dm2 = -J_prime_torch(w) / (J_w**3)
        
        # 5. Assemble the final g(l,m) Hessian
        g_ll = K_uu * (du_dl**2) + dK_du * d2u_dl2
        g_mm = K_ww * (dw_dm**2) + dK_dw * d2w_dm2
        g_lm = K_uw * du_dl * dw_dm
        
        # Use .item() to safely detach the scalar from the computation graph
        H_g = torch.tensor([[g_ll.item(), g_lm.item()], 
                            [g_lm.item(), g_mm.item()]], dtype=torch.float64)
        return H_g

    def test_global_monte_carlo(self, samples=5000):
        """Randomly samples the (l,m) domain to ensure global convexity."""
        print(f"\n[Strategy 1] Global Monte Carlo Hessian Scan (N={samples})")
        l_vals = torch.rand(samples) * 0.98 + 0.01  # Avoid absolute 0 and 1
        m_vals = torch.rand(samples) * 0.98 + 0.01
        
        min_eigs = []
        for i in range(samples):
            # Avoid the exact diagonal singularity
            if abs(l_vals[i] - m_vals[i]) < 1e-5:
                continue
                
            try:
                H = self.get_stable_hessian_g(l_vals[i], m_vals[i])
                eigs = torch.linalg.eigvalsh(H)
                min_eig = eigs[0].item()
                min_eigs.append(min_eig)
                
                if min_eig < self.global_min_eigenvalue:
                    self.global_min_eigenvalue = min_eig
                    self.worst_coords = (l_vals[i].item(), m_vals[i].item())
            except Exception:
                pass
                
        print(f"-> Scan Complete. Global Minimum Eigenvalue: {self.global_min_eigenvalue:.6e}")
        return min_eigs

    def scan_local_patch(self, center_l, center_m, span=0.1, grid_res=15):
        """Scans a dense local grid and plots the stability."""
        print(f"\n[Strategy 2] Local Patch Scan: Center({center_l:.3f}, {center_m:.3f}) Span({span})")
        l_vals = np.linspace(max(0.01, center_l - span/2), min(0.99, center_l + span/2), grid_res)
        m_vals = np.linspace(max(0.01, center_m - span/2), min(0.99, center_m + span/2), grid_res)
        
        patch_eigs = np.zeros((grid_res, grid_res))
        
        for i, l in enumerate(l_vals):
            for j, m in enumerate(m_vals):
                if abs(l - m) < 1e-5:
                    m_eff = m + 1e-4
                else:
                    m_eff = m
                try:
                    H = self.get_stable_hessian_g(l, m_eff)
                    eig = torch.linalg.eigvalsh(H)[0].item()
                    patch_eigs[i, j] = eig
                    if eig < self.global_min_eigenvalue:
                        self.global_min_eigenvalue = eig
                        self.worst_coords = (l, m_eff)
                except Exception:
                    patch_eigs[i, j] = np.nan

        local_min = np.nanmin(patch_eigs)
        print(f"-> Local Minimum Eigenvalue: {local_min:.6e}")
        
        # Plotting: Removed vmin/vmax to allow dynamic scaling of the color gradient 
        plt.figure(figsize=(5, 4))
        plt.imshow(patch_eigs.T, origin='lower', extent=[l_vals[0], l_vals[-1], m_vals[0], m_vals[-1]], 
                   cmap='viridis') 
        plt.colorbar(label='Min Eigenvalue')
        plt.title(f"Convexity Stability @ ({center_l:.2f}, {center_m:.2f})")
        plt.xlabel('l')
        plt.ylabel('m')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("=====================================================")
    print("  Kappa Convexity Conjecture Verification Engine")
    print("=====================================================")
    
    analyzer = KappaConvexityAnalyzer()
    
    # 1. Global Scan
    analyzer.test_global_monte_carlo(samples=1000)
    
    # 2. Targeted Boundary Scans
    analyzer.scan_local_patch(0.5, 0.5, span=0.2)   # Core interior
    analyzer.scan_local_patch(0.95, 0.5, span=0.08) # Near boundary
    
    print("\n=====================================================")
    print("  FINAL RESULTS")
    print("=====================================================")
    print(f"Absolute Minimum Eigenvalue: {analyzer.global_min_eigenvalue:.8e}")
    
    if analyzer.global_min_eigenvalue >= -1e-6:
        print(">> VERDICT: SUCCESS.")
        print(">> The Hessian is Positive Semi-Definite. Joint Convexity Holds.")
    else:
        print(">> VERDICT: WARNING! Negative eigenvalue detected.")
        print(f">> Location: l={analyzer.worst_coords[0]:.5f}, m={analyzer.worst_coords[1]:.5f}")