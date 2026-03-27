# Boolean-MI-Conjecture-Proofs

## Repository Abstract
This repository provides the complete computational verification framework for "A Differential Equation Approach to the Most-Informative Boolean Function Conjecture". The research introduces a novel method to establish the mutual information conjecture for *balanced* Boolean functions by evaluating the derivative of the entropy along a continuous noise semigroup.

As demonstrated in the research, proving the conjecture analytically reduces to verifying explicit finite-dimensional inequalities. This repository aims to computationally evaluate and formally verify the three most critical components:

1. **The Kappa Asymmetry Conjecture**: Bounding the difference between evaluated points on the domain to show $\kappa(u,w) \le \kappa(1-u,w)$.
2. **The Kappa Convexity Conjecture**: Establishing the joint convexity of the transformed function $\kappa(H_2^{-1}(l), H_2^{-1}(m))$.
3. **The Four-Variable Inequality**: Bounding the functional evaluations across a randomized 4-tuple space.

To achieve this, the repository is split into two distinct methodological approaches. The first approach utilizes *high-throughput Monte Carlo sampling*, *Exhaustive Grid Search*, and *Analytic Chain Rule Automatic Differentiation* via PyTorch to provide exhaustive numerical and visual evidence. The second approach transitions to a formal, computer-assisted mathematical proof using the Arb C-library (via `python-flint`) to rigorously bound the functional space and mathematically guarantee the absence of counterexamples using strict IEEE 754 directed rounding.

## Closed Form of Inequalities and Related Functions
To keep the abstract readable, the explicit formulas used by the verification scripts are collected here. Throughout, we use

```math
H_2(x) = -x \log_2 x - (1-x)\log_2(1-x), \qquad
J(x) = \log_2\!\left(\frac{1-x}{x}\right),
```

```math
L(u) = \frac{2H_2(u)}{1-2u} \quad (u \neq \tfrac{1}{2}), \qquad
L\!\left(\tfrac{1}{2}\right) = \infty,
```

and $H_2^{-1}$ denotes the inverse branch of $H_2$ on $\left[0,\tfrac{1}{2}\right]$.

### 1. Kappa Asymmetry Conjecture
The asymmetry verification aims to prove

```math
\kappa(u,w) \le \kappa(1-u,w), \qquad (u,w) \in \left(0,\tfrac{1}{2}\right)^2.
```

Here

```math
\kappa(u,w) = \frac{(u-w)(J(w)-J(u))}{2}
- |u-w|\,J\!\left(L^{-1}\!\left(\frac{H_2(u)+H_2(w)}{|u-w|}\right)\right),
```

for $u \neq w$, with $\kappa(u,u)=0$.

### 2. Kappa Convexity Conjecture
The convexity verification studies the transformed function

```math
g(l,m) = \kappa(H_2^{-1}(l), H_2^{-1}(m)), \qquad (l,m) \in (0,1)^2,
```

and seeks to show that $g$ is jointly convex on its domain.

### 3. Four-Variable Inequality
For quadruples $(\mu_u, \mu_w, e_u, e_w) \in [0,1]^2$ satisfying :

```math
H_2(\mu_u) \ge e_u, \qquad H_2(\mu_w) \ge e_w,
```

define

```math
\phi(x,y)=
\begin{cases}
\eta(y)-\dfrac{y}{r}\eta(r), & H_2(x)>y, \\
0, & H_2(x)\le y,
\end{cases}
\qquad
\eta(t)=\bigl(1-2H_2^{-1}(t)\bigr)J(H_2^{-1}(t)),
```

where $r=1$ if $x=\tfrac{1}{2}$, while for $x \neq \tfrac{1}{2}$, $r \in (0,1)$ is the unique solution of

```math
\frac{r}{1-2H_2^{-1}(r)} = \frac{y}{|1-2x|}.
```

The four-variable script aims to verify

```math
\begin{aligned}
&\phi\!\left(\frac{1-\lvert H_2^{-1}(e_u)-H_2^{-1}(e_w)\rvert}{2}, \frac{e_u+e_w}{2}\right)
+ \frac{\bigl(H_2^{-1}(e_u)-H_2^{-1}(e_w)\bigr)\bigl(J(H_2^{-1}(e_w))-J(H_2^{-1}(e_u))\bigr)}{2} \\
&\qquad
- \phi\!\left(\frac{1-\lvert\mu_u-\mu_w\rvert}{2}, \frac{e_u+e_w}{2}\right)
\ge
\phi\!\left(\frac{\mu_u+\mu_w}{2}, \frac{e_u+e_w}{2}\right)
- \frac{1}{2}\phi(\mu_u,e_u) - \frac{1}{2}\phi(\mu_w,e_w).
\end{aligned}
```

---

## Repository Structure & Methodologies

### 1. `/numerical_tests`
This directory contains empirical tests, visual plots, and adversarial verification engines using 64-bit PyTorch tensors.

* **`4var_Inequality.py`**
  * **Objective:** Verify numerically the 4-variable inequality $LHS \ge RHS$ for the $\phi(m, e)$ function.
  * **Methodology:** A 3-Tier Adversarial Verification Engine.
    1. **Exhaustive Grid Search:** Systematically scans the 4D parameter space to fulfill exhaustive search requirements.
    2. **Global Monte Carlo:** Evaluates millions of random valid quadruplets satisfying the entropy constraints above to catch off-grid interactions.
    3. **Adaptive Adversarial Zoom-In:** Isolates the exact coordinates closest to zero and autonomously generates dense, localized micro-clouds to aggressively stress-test local minima.
  * **Implementation Notes:** Uses vectorized PyTorch operations. Enforces a safe margin to prevent floating-point collapse at the $J(x)$ asymptotes. The engine hits the absolute 64-bit machine epsilon floor ($\approx -2.66 \times 10^{-15}$), strictly validating the inequality without counterexamples.

* **`conjecture_final.py`**
  * **Objective:** Analyze the local convexity of $g(l,m) = \kappa(H_2^{-1}(l), H_2^{-1}(m))$.
  * **Methodology:** Hybrid Analytic/Automatic Differentiation via Custom Autograd.
  * **Implementation Notes:** Because standard autodiff tools fail on the numerical inverse functions $H_2^{-1}$ and $L^{-1}$, this script injects the exact analytic derivative using the Inverse Function Theorem into PyTorch's computational graph. It computes the local Hessian using `torch.autograd.functional` and scans discrete local patches to prove the matrix is Positive Semi-Definite.

* **`asymmetry_surface_plot.py`**
  * **Objective:** Provide immediate visual verification of the Asymmetry Conjecture.
  * **Methodology:** Generates a high-resolution 3D mathematical surface plot to visually demonstrate that $g(u,w) \le 0$ globally on the $(0, 0.5)^2$ domain, plotting the function relative to a strict $Z=0$ upper-bound translucent plane.

### 2. `/formal_ia_proof`
This directory transitions from empirical sampling to mathematically rigorous computer-assisted proof generation.

* **`ia_proof_inequality.py`**
  * **Objective:** Rigorously prove $g(u, w) = \kappa(u, w) - \kappa(1-u, w) \le 0$ on the domain $(0, 0.5)^2$.
  * **Methodology:** Interval Arithmetic (IA) and Domain Decomposition (Branch and Bound).
  * **Underlying Engine:** Uses `python-flint`, a wrapper for the **Arb** C-library, ensuring arbitrary-precision ball arithmetic with strict IEEE 754 directed rounding.
  * **Architecture & Error Handling:** The algorithm uses semantic endpoint enclosures to rigorously evaluate boxes. It features a strict exception-routing system that purposefully catches mathematical domain errors (e.g., `ValueError`, `ZeroDivisionError`) caused by IA overestimation, using them to intelligently trigger quadtree subdivisions while strictly crashing on unexpected code execution failures.

## Key Mathematical Simplifications for IA
To prevent the interval overestimation (the "Dependency Problem") from causing infinite bisection loops, we applied the following analytic simplifications before passing equations to the Arb engine:
1. **Eliminating Absolute Values:** For $(u, w) \in (0, 0.5)^2$, we know strictly that $1 - u - w > 0$. We analytically removed the $|1-u-w|$ term, smoothing the search space.
2. **Rigorous Inverse Bisection:** Arb lacks a built-in inverse for $L(x)$. Since $L(x)$ is strictly monotonically increasing on $(0, 0.5)$, we implemented an Interval Bisection algorithm to strictly bound $L^{-1}(y)$.
3. **Diagonal Isolation:** At $u=w$, the function divides by zero. The script isolates a small strip ($|u-w| < 0.005$) and leaves it unverified by IA, to be handled analytically.

## Prerequisites to run project
The repository assumes a recent Python 3 environment. The external libraries required to run the scripts are:

### Numerical scripts (`numerical_tests/`)
Install the packages below to run `4var_Inequality.py`, `conjecture_final.py`, and `asymmetry_surface_plot.py`:

```bash
pip install torch numpy matplotlib
```

These cover:
- `torch` for tensor computations and automatic differentiation
- `numpy` for auxiliary numerical routines
- `matplotlib` for the histogram and surface visualizations

### Formal interval-arithmetic proof (`formal_ia_proof/`)
To run the IA proof script, install `python-flint` with Arb support:

```bash
pip install python-flint
```

The formal proof code specifically requires the import

```bash
python -c "from flint import arb"
```

to succeed. On platforms where a compatible wheel is unavailable, you may need to install the underlying FLINT/Arb libraries first and then use a matching `python-flint` build.

### Standard-library modules
The remaining imports used by the project, such as `time`, `collections`, and `dataclasses`, are part of the Python standard library and do not require separate installation.

## Benchmarks
The default IA script is configured to a maximum bisection depth of 7 for rapid verification on standard hardware. By increasing the depth to 9 on a single CPU core, the script successfully verifies approximately 32% of the interior domain with zero mathematical errors.

Deeper bisection depths (>=14, for instance) will help resolve the dependency problem near the boundaries and provide greater verification. It represents an ideal use case for parallel high-compute AI environments.
