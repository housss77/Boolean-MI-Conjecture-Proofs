"""
Arb certificate for the candidate four-variable counterexample.

This script does not use torch, floats, random search, or tolerance-based
acceptance.  It uses python-flint's Arb ball arithmetic and monotone inverse
bracketing to certify the sign of LHS - RHS for the tuple

    (mu, mw, eu, ew) = (6.6e-7, 4.0e-7, 1.1e-5, 9.0e-6).

If the final printed upper bound for LHS - RHS is strictly below zero, the
candidate is a rigorous counterexample to the inequality as implemented here
and as stated in the repository README, subject to the usual trust assumption
on Arb/python-flint.
"""

from __future__ import annotations

from dataclasses import dataclass

from flint import arb, ctx


ctx.prec = 1536

ZERO = arb("0")
ONE = arb("1")
TWO = arb("2")
HALF = arb("0.5")

INVERSE_BISECTIONS = 620


def lower(x: arb) -> arb:
    return x.lower()


def upper(x: arb) -> arb:
    return x.upper()


def certified_positive(x: arb) -> bool:
    return lower(x) > ZERO


def certified_negative(x: arb) -> bool:
    return upper(x) < ZERO


def certified_less(x: arb, y: arb) -> bool:
    return upper(x) < lower(y)


def certified_greater(x: arb, y: arb) -> bool:
    return lower(x) > upper(y)


def require_positive(name: str, x: arb) -> None:
    if not certified_positive(x):
        raise RuntimeError(f"Could not certify positivity of {name}: {x}")


def require_negative(name: str, x: arb) -> None:
    if not certified_negative(x):
        raise RuntimeError(f"Could not certify negativity of {name}: {x}")


def require_interval_inside(name: str, x: arb, lo: arb, hi: arb) -> None:
    if not (lower(x) > lo and upper(x) < hi):
        raise RuntimeError(f"Could not certify {name} lies in ({lo}, {hi}): {x}")


def log2(x: arb) -> arb:
    return x.log() / TWO.log()


def H2(x: arb) -> arb:
    require_interval_inside("H2 argument", x, ZERO, ONE)
    return -(x * log2(x) + (ONE - x) * log2(ONE - x))


def J(x: arb) -> arb:
    require_interval_inside("J argument", x, ZERO, ONE)
    return log2((ONE - x) / x)


def L(u: arb) -> arb:
    require_interval_inside("L argument", u, ZERO, HALF)
    return TWO * H2(u) / (ONE - TWO * u)


def interval_from_bounds(lo: arb, hi: arb) -> arb:
    if not lower(lo) <= upper(hi):
        raise RuntimeError(f"Invalid interval bounds: lo={lo}, hi={hi}")
    return lo.union(hi)


def inverse_increasing(
    func,
    y: arb,
    *,
    lo: arb = ZERO,
    hi: arb = HALF,
    iterations: int = INVERSE_BISECTIONS,
    label: str,
) -> arb:
    """
    Rigorously enclose func^{-1}(y) for increasing func on (lo, hi).

    The maintained invariant is that the true inverse set is contained in the
    current [left, right] bracket.  We only move a bracket endpoint after a
    certified Arb comparison.  If a comparison becomes undecidable because the
    midpoint image overlaps y, the current bracket is already a valid enclosure.
    """
    left = lo
    right = hi

    for _ in range(iterations):
        mid = (left + right) / TWO
        f_mid = func(mid)

        if certified_less(f_mid, y):
            left = mid
        elif certified_greater(f_mid, y):
            right = mid
        else:
            break

    result = interval_from_bounds(left, right)
    require_interval_inside(label, result, lo, hi)
    return result


def H2_inv(y: arb) -> arb:
    require_interval_inside("H2_inv target", y, ZERO, ONE)
    return inverse_increasing(H2, y, label="H2_inv result")


def L_inv(y: arb) -> arb:
    require_positive("L_inv target", y)
    return inverse_increasing(L, y, label="L_inv result")


def eta_from_u(u: arb) -> arb:
    return (ONE - TWO * u) * J(u)


def eta_from_entropy(z: arb) -> arb:
    return eta_from_u(H2_inv(z))


def phi(m: arb, e: arb, name: str) -> arb:
    """
    Certified evaluation of phi(m, e) on the active branch.

    For this candidate, every phi call used by the inequality is strictly on
    the active branch H2(m) > e.  If this cannot be certified, the script stops.
    """
    require_interval_inside(f"{name}.m", m, ZERO, ONE)
    require_interval_inside(f"{name}.e", e, ZERO, ONE)

    branch_margin = H2(m) - e
    require_positive(f"{name}: H2(m)-e", branch_margin)

    denominator = abs(ONE - TWO * m)
    require_positive(f"{name}: |1-2m|", denominator)

    target = TWO * e / denominator
    u_r = L_inv(target)
    r = H2(u_r)
    require_positive(f"{name}: r", r)

    return eta_from_entropy(e) - (e / r) * eta_from_u(u_r)


@dataclass
class Certificate:
    domain_mu: arb
    domain_mw: arb
    lhs: arb
    rhs: arb
    diff: arb


def compute_certificate() -> Certificate:
    mu = arb("6.6e-7")
    mw = arb("4.0e-7")
    eu = arb("1.1e-5")
    ew = arb("9.0e-6")

    domain_mu = H2(mu) - eu
    domain_mw = H2(mw) - ew
    require_positive("H2(mu)-eu", domain_mu)
    require_positive("H2(mw)-ew", domain_mw)

    e_avg = (eu + ew) / TWO

    u_eu = H2_inv(eu)
    u_ew = H2_inv(ew)
    delta_entropy_inv = u_eu - u_ew
    require_positive("H2_inv(eu)-H2_inv(ew)", delta_entropy_inv)

    delta_mu = mu - mw
    require_positive("mu-mw", delta_mu)

    arg1_m = (ONE - delta_entropy_inv) / TWO
    arg3_m = (ONE - delta_mu) / TWO
    avg_m = (mu + mw) / TWO

    term1 = phi(arg1_m, e_avg, "term1_phi")
    term2 = delta_entropy_inv * (J(u_ew) - J(u_eu)) / TWO
    term3 = -phi(arg3_m, e_avg, "term3_phi")

    lhs = term1 + term2 + term3
    rhs = (
        phi(avg_m, e_avg, "rhs_avg_phi")
        - phi(mu, eu, "rhs_mu_phi") / TWO
        - phi(mw, ew, "rhs_mw_phi") / TWO
    )
    diff = lhs - rhs

    require_negative("LHS-RHS", diff)
    return Certificate(domain_mu=domain_mu, domain_mw=domain_mw, lhs=lhs, rhs=rhs, diff=diff)


def print_interval(name: str, x: arb) -> None:
    print(f"{name}:")
    print(f"  enclosure = {x}")
    print(f"  lower     = {lower(x)}")
    print(f"  upper     = {upper(x)}")


if __name__ == "__main__":
    cert = compute_certificate()

    print("Certified candidate:")
    print("  mu = 6.6e-7")
    print("  mw = 4.0e-7")
    print("  eu = 1.1e-5")
    print("  ew = 9.0e-6")
    print()

    print_interval("H2(mu) - eu", cert.domain_mu)
    print_interval("H2(mw) - ew", cert.domain_mw)
    print_interval("LHS", cert.lhs)
    print_interval("RHS", cert.rhs)
    print_interval("LHS - RHS", cert.diff)
    print()

    print("Conclusion:")
    print("  H2(mu) - eu is certified positive.")
    print("  H2(mw) - ew is certified positive.")
    print("  LHS - RHS is certified negative.")
    print("  Therefore the tuple is a certified counterexample to the stated inequality.")
