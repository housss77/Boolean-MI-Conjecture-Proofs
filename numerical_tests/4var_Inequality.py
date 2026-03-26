import torch
import numpy as np
import matplotlib.pyplot as plt

# Use float64 for better numerical stability.
torch.set_default_dtype(torch.float64)
EPS = 1e-12

# --- Core Math Functions ---

def H2(x):
    x = torch.clamp(x, EPS, 1 - EPS)
    return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

def J(x):
    x = torch.clamp(x, EPS, 1 - EPS)
    return torch.log2((1 - x) / x)

def L(u):
    u = torch.clamp(u, EPS, 0.5 - EPS)
    return 2 * H2(u) / (1 - 2 * u)

def Linv(y):
    lo = torch.zeros_like(y) + EPS
    hi = torch.zeros_like(y) + 0.5 - EPS
    for _ in range(60):
        mid = (lo + hi) / 2
        val = L(mid)
        mask = val < y
        lo = torch.where(mask, mid, lo)
        hi = torch.where(mask, hi, mid)
    return (lo + hi) / 2

def H2inv(y):
    lo = torch.zeros_like(y) + EPS
    hi = torch.zeros_like(y) + 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        val = H2(mid)
        mask = val < y
        lo = torch.where(mask, mid, lo)
        hi = torch.where(mask, hi, mid)
    return (lo + hi) / 2

# --- The Phi Function Implementation ---

def eta(z):
    u = H2inv(z)
    return (1 - 2 * u) * J(u)

def phi(m, e):
    m = torch.as_tensor(m)
    e = torch.as_tensor(e)
    h2_m = H2(m)
    active_mask = h2_m > e + EPS
    result = torch.zeros_like(m)
    
    if active_mask.any():
        m_active = m[active_mask]
        e_active = e[active_mask]
        numerator = 2 * e_active
        denominator = torch.abs(1 - 2 * m_active)
        denominator = torch.where(denominator < EPS, torch.ones_like(denominator)*EPS, denominator)
        
        target_L = numerator / denominator
        u_r = Linv(target_L)
        r = H2(u_r)
        
        term1 = eta(e_active)
        term2 = (e_active / r) * eta(r)
        term2 = torch.where(r < EPS, torch.zeros_like(r), term2)
        
        result[active_mask] = term1 - term2
    return result

# --- The Inequality Check ---

def inequality_diff(mu, mw, eu, ew):
    u_eu = H2inv(eu)
    u_ew = H2inv(ew)
    
    arg1_m = (1 - torch.abs(u_eu - u_ew)) / 2
    arg1_e = (eu + ew) / 2
    term1 = phi(arg1_m, arg1_e)
    
    term2 = (u_eu - u_ew) * (J(u_ew) - J(u_eu)) / 2
    
    arg3_m = (1 - torch.abs(mu - mw)) / 2
    arg3_e = (eu + ew) / 2
    term3 = -phi(arg3_m, arg3_e)
    
    rhs_term1 = phi((mu + mw)/2, (eu + ew)/2)
    rhs_term2 = -0.5 * phi(mu, eu)
    rhs_term3 = -0.5 * phi(mw, ew)
    
    return (term1 + term2 + term3) - (rhs_term1 + rhs_term2 + rhs_term3)

# --- ADVANCED VERIFICATION ENGINE ---

class FourVarTester:
    def __init__(self):
        self.global_min = float('inf')
        self.worst_case_coords = None

    def update_min(self, diff, mu, mw, eu, ew):
        current_min = torch.min(diff).item()
        if current_min < self.global_min:
            self.global_min = current_min
            min_idx = torch.argmin(diff).item()
            self.worst_case_coords = (mu[min_idx].item(), mw[min_idx].item(), eu[min_idx].item(), ew[min_idx].item())

    def test_grid_search(self, resolution=30):
        print(f"\n[Strategy 1] Exhaustive Grid Search (Resolution: {resolution}^4)...")
        m_vals = torch.linspace(0.01, 0.99, resolution)
        
        # Construct valid pairs where H2(m) >= e
        mu_grid, mw_grid, eu_grid, ew_grid = [], [], [], []
        
        for mu in m_vals:
            for mw in m_vals:
                # e must be between 0 and H2(m)
                e_u_vals = torch.linspace(0.001, H2(mu).item() - 0.001, resolution//2)
                e_w_vals = torch.linspace(0.001, H2(mw).item() - 0.001, resolution//2)
                
                for eu in e_u_vals:
                    for ew in e_w_vals:
                        mu_grid.append(mu)
                        mw_grid.append(mw)
                        eu_grid.append(eu)
                        ew_grid.append(ew)
                        
        mu_t = torch.tensor(mu_grid)
        mw_t = torch.tensor(mw_grid)
        eu_t = torch.tensor(eu_grid)
        ew_t = torch.tensor(ew_grid)
        
        diff = inequality_diff(mu_t, mw_t, eu_t, ew_t)
        self.update_min(diff, mu_t, mw_t, eu_t, ew_t)
        print(f"-> Grid Search Complete. Tested {len(mu_t)} points. Current Min: {self.global_min:.8e}")

    def test_monte_carlo(self, batch_size=1_000_000):
        print(f"\n[Strategy 2] Global Monte Carlo Random Sampling (N={batch_size})...")
        mu = torch.rand(batch_size)
        eu = torch.rand(batch_size) * H2(mu)
        
        mw = torch.rand(batch_size)
        ew = torch.rand(batch_size) * H2(mw)
        
        diff = inequality_diff(mu, mw, eu, ew)
        self.update_min(diff, mu, mw, eu, ew)
        
        # Save the worst 1000 points for adaptive search
        k = min(1000, batch_size)
        worst_diffs, worst_indices = torch.topk(diff, k, largest=False)
        self.suspect_points = (mu[worst_indices], mw[worst_indices], eu[worst_indices], ew[worst_indices])
        
        print(f"-> Monte Carlo Complete. Current Min: {self.global_min:.8e}")
        return diff

    def test_adversarial_zoom(self, spread=0.01, samples_per_point=1000):
        print(f"\n[Strategy 3] Adaptive Adversarial Zoom-In...")
        print(f"-> Targeting the {len(self.suspect_points[0])} most vulnerable regions from Strategy 2.")
        
        mu_base, mw_base, eu_base, ew_base = self.suspect_points
        
        # We enforce a safe margin to prevent floating-point collapse 
        # at the J(x) singularities (x=0, x=1)
        SAFE_MARGIN = 1e-4
        
        # Expand around the worst points with a normal distribution, staying off the absolute edge
        mu_raw = mu_base.repeat(samples_per_point) + torch.randn(len(mu_base)*samples_per_point) * spread
        mu_zoom = torch.clamp(mu_raw, SAFE_MARGIN, 1 - SAFE_MARGIN)
        
        mw_raw = mw_base.repeat(samples_per_point) + torch.randn(len(mw_base)*samples_per_point) * spread
        mw_zoom = torch.clamp(mw_raw, SAFE_MARGIN, 1 - SAFE_MARGIN)
        
        # Ensure e is still valid under the new random m
        eu_raw = eu_base.repeat(samples_per_point) + torch.randn(len(eu_base)*samples_per_point) * spread
        eu_zoom = torch.minimum(eu_raw.clamp_min(SAFE_MARGIN), H2(mu_zoom))
        
        ew_raw = ew_base.repeat(samples_per_point) + torch.randn(len(ew_base)*samples_per_point) * spread
        ew_zoom = torch.minimum(ew_raw.clamp_min(SAFE_MARGIN), H2(mw_zoom))
        
        diff = inequality_diff(mu_zoom, mw_zoom, eu_zoom, ew_zoom)
        self.update_min(diff, mu_zoom, mw_zoom, eu_zoom, ew_zoom)
        print(f"-> Adversarial Zoom Complete. Tested {len(mu_zoom)} adaptive points. Final Min: {self.global_min:.8e}")


if __name__ == "__main__":
    print("=====================================================")
    print("  4-Variable Inequality Verification Engine")
    print("=====================================================")
    
    tester = FourVarTester()
    
    # Run the testing suite
    tester.test_grid_search(resolution=25)
    diff_mc = tester.test_monte_carlo(batch_size=2_000_000)
    tester.test_adversarial_zoom(spread=0.005, samples_per_point=2000)
    
    print("\n=====================================================")
    print("  FINAL RESULTS")
    print("=====================================================")
    print(f"Absolute Minimum (LHS - RHS): {tester.global_min:.10f}")
    
    if tester.global_min >= -1e-9:
        print("\n>> VERDICT: SUCCESS.")
        print(">> No counterexamples found across Grid, Monte Carlo, and Adversarial Adaptive searches.")
    else:
        print("\n>> VERDICT: WARNING! Potential counterexample found.")
        c = tester.worst_case_coords
        print(f">> Coordinates: mu={c[0]:.5f}, mw={c[1]:.5f}, eu={c[2]:.5f}, ew={c[3]:.5f}")

    # Visualizing the Monte Carlo Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(diff_mc.numpy(), bins=150, log=True, color='#2ca02c', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Falsification Boundary (Zero)')
    plt.title("Distribution of (LHS - RHS) via Global Monte Carlo", fontsize=14)
    plt.xlabel("Difference Value (LHS - RHS)", fontsize=12)
    plt.ylabel("Frequency Count (Log Scale)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()