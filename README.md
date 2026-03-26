# Boolean-MI-Conjecture-Proofs

## Repository Abstract
This repository provides the complete computational verification framework for "A Differential Equation Approach to the Most-Informative Boolean Function Conjecture" (Chen, Gohari, Nair, ISIT 2025). The research introduces a novel method to establish the mutual information conjecture for balanced Boolean functions by evaluating the derivative of the entropy along a continuous noise semigroup. 

As demonstrated in the paper, proving the conjecture analytically reduces to verifying four explicit finite-dimensional inequalities. This repository contains the code required to computationally evaluate and formally verify the three most critical components:
1. **The Kappa Asymmetry Conjecture**: Bounding the difference between evaluated points on the domain.
2. **The Kappa Convexity Conjecture**: Establishing the joint convexity of the transformed function.
3. **The Four-Variable Inequality**: Bounding the functional evaluations across a randomized 4-tuple space.

To achieve this, the repository is split into two distinct methodological approaches. The first approach utilizes high-throughput Monte Carlo sampling, Exhaustive Grid Search, and Analytic Chain Rule Automatic Differentiation via PyTorch to provide exhaustive numerical and visual evidence. The second approach transitions to a formal, computer-assisted mathematical proof using the Arb C-library (via `python-flint`) to rigorously bound the functional space and mathematically guarantee the absence of counterexamples using strict IEEE 754 directed rounding.

---

## Repository Structure & Methodologies

### 1. `/numerical_exploration`
This directory contains empirical tests and adversarial verification engines using 64-bit PyTorch tensors.

* **`4var_Inequality.py`**
  * **Objective:** Verify the 4-variable inequality $LHS \ge RHS$ for the $\phi(m, e)$ function.
  * **Methodology:** A 3-Tier Adversarial Verification Engine.
    1. **Exhaustive Grid Search:** Systematically scans the 4D parameter space to fulfill exhaustive search requirements.
    2. **Global Monte Carlo:** Evaluates millions of random valid quadruplets $(\mu, \mu_w, e_u, e_w)$ where $H_2(m) \ge e$ to catch off-grid interactions.
    3. **Adaptive Adversarial Zoom-In:** Isolates the exact coordinates closest to zero and autonomously generates dense, localized micro-clouds to aggressively stress-test local minima.
  * **Implementation Notes:** Uses vectorized PyTorch operations. Enforces a safe margin to prevent floating-point collapse at the $J(x)$ asymptotes. The engine hits the absolute 64-bit machine epsilon floor ($\approx -2.66 \times 10^{-15}$), strictly validating the inequality without counterexamples.

* **`conjecture_final.py`**
  * **Objective:** Analyze the local convexity/concavity of $g(l,m) = \kappa(u(l), w(m))$.
  * **Methodology:** Hybrid Analytic/Automatic Differentiation. 
  * **Implementation Notes:** Because standard autodiff tools fail on the numerical inverse functions $H_2^{-1}$ and $L^{-1}$, this script computes the exact local Hessian using the Analytic Chain Rule combined with `torch.func.hessian` and `torch.func.jacrev`. It scans discrete local patches to find minimum eigenvalues.

### 2. `/formal_ia_proof`
This directory transitions from empirical sampling to mathematically rigorous computer-assisted proof generation.

* **`proof-2.py`**
  * **Objective:** Rigorously prove $g(u, w) = \kappa(u, w) - \kappa(1-u, w) \le 0$ on the domain $(0, 0.5)^2$.
  * **Methodology:** Interval Arithmetic (IA) and Domain Decomposition (Branch and Bound).
  * **Underlying Engine:** Uses `python-flint`, a wrapper for the **Arb** C-library, ensuring arbitrary-precision ball arithmetic with strict IEEE 754 directed rounding.
  * **Algorithm:** The domain is placed in a queue. For each box $U \times W$, Arb computes strict bounds for $g(U,W)$. If the upper bound $\le 0$, the box is verified. If the interval straddles zero, the box is bisected (Skelboe-Moore algorithm).

## Key Mathematical Simplifications for IA
To prevent the interval overestimation (the "Dependency Problem") from causing infinite bisection loops, we applied the following analytic simplifications before passing equations to the Arb engine:
1. **Eliminating Absolute Values:** For $(u, w) \in (0, 0.5)^2$, we know strictly that $1 - u - w > 0$. We analytically removed the $|1-u-w|$ term, smoothing the search space.
2. **Rigorous Inverse Bisection:** Arb lacks a built-in inverse for $L(x)$. Since $L(x)$ is strictly monotonically increasing on $(0, 0.5)$, we implemented a rigorous Interval Bisection algorithm to strictly bound $L^{-1}(y)$.
3. **Diagonal Isolation:** At $u=w$, the function divides by zero. The script isolates a narrow strip ($|u-w| < 0.005$) and leaves it unverified by IA, to be handled analytically.

## Benchmarks
At a maximum bisection depth of 9 on a single CPU core, `proof-2.py` successfully verifies approximately 32% of the interior domain with zero mathematical errors. Deeper bisection depths (14+) are required to resolve the dependency problem near the boundaries, representing an ideal use case for massively parallel, high-compute environments.
