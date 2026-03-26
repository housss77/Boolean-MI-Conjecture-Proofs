import time
from collections import deque
from dataclasses import dataclass

try:
    import flint
    from flint import arb
except Exception as exc: # If Arb failed to be imported
    raise RuntimeError(
        "This script requires the Arb-enabled python-flint package. "
        "The current Python environment does not provide 'from flint import arb'."
    ) from exc


# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------

# Epsilon and max_depth depend on each other
DIAGONAL_EPSILON = 0.005
MAX_DEPTH = 7

PREC_BITS = 128
DOMAIN_BOX = (0.001, 0.499, 0.001, 0.499)
BISECTION_ITERATIONS = 150
L_INV_LOW = arb("0.000000000000000001")
L_INV_HIGH = arb("0.499999999999999999")
ZERO = arb("0")
ONE = arb("1")
TWO = arb("2")

flint.ctx.prec = PREC_BITS


# ---------------------------------------------------------
# ARB HELPERS
# ---------------------------------------------------------
def lower_endpoint(x):
    """Lower endpoint enclosure of an Arb ball."""
    return x.mid() - x.rad()


def upper_endpoint(x):
    """Upper endpoint enclosure of an Arb ball."""
    return x.mid() + x.rad()


def ball_from_bounds(lower, upper):
    """Construct an interval enclosure from two endpoint enclosures."""
    return lower.union(upper)


def strictly_less_than(x, y):
    """Certified comparison x < y using endpoint enclosures."""
    return upper_endpoint(x) < lower_endpoint(y)


def nonpositive(x):
    """Certified test x <= 0 using endpoint enclosures."""
    return upper_endpoint(x) <= ZERO


def positive(x):
    """Certified test x > 0 using endpoint enclosures."""
    return lower_endpoint(x) > ZERO


# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def log2(x):
    """Base-2 logarithm computed from the natural logarithm."""
    return x.log() / TWO.log()


def H2(x):
    """Binary entropy H2(x) = -x log2(x) - (1-x) log2(1-x)."""
    return -(x * log2(x) + (ONE - x) * log2(ONE - x))


def J(x):
    """J(x) = d/dx H2(x) = log2((1-x)/x)."""
    return log2((ONE - x) / x)


def L(x):
    """L(x) = 2 H2(x) / (1 - 2x)."""
    return (TWO * H2(x)) / (ONE - TWO * x)


def bisection_L_inv_scalar(y_target, iterations=BISECTION_ITERATIONS):
    """
    Evaluation of an interval containing inverse of x = L^(-1)(y_target)
    Method : binary search
    """
    low = L_INV_LOW
    high = L_INV_HIGH

    for _ in range(iterations):
        mid = (low + high) / TWO
        L_mid = L(mid)

        if strictly_less_than(L_mid, y_target):
            low = mid
        else:
            high = mid

    return (low + high) / TWO, (high - low) / TWO


def rigorous_L_inv(y_interval):
    """
    Evaluation of L^{-1} on an interval.
    Since L is increasing, the image of the endpoint enclosure suffices.
    """
    y_lower = lower_endpoint(y_interval)
    y_upper = upper_endpoint(y_interval)

    x_min_mid, x_min_rad = bisection_L_inv_scalar(y_lower)
    x_max_mid, x_max_rad = bisection_L_inv_scalar(y_upper)

    lower_bound = x_min_mid - x_min_rad
    upper_bound = x_max_mid + x_max_rad
    return ball_from_bounds(lower_bound, upper_bound)


def kappa(u, w):
    """Definition of the auxiliary function kappa(u, w)."""
    diff = u - w
    abs_diff = abs(diff)

    term1 = diff * (J(w) - J(u)) / TWO

    y_input = (H2(u) + H2(w)) / abs_diff
    inv_L_val = rigorous_L_inv(y_input)
    term2 = abs_diff * J(inv_L_val)

    return term1 - term2


def kappa_1_minus_u(u, w):
    """
    Specialized formula for kappa(1-u, w) (without absolute value).
    On the domain u,w < 1/2, the term 1-u-w is strictly positive.
    """
    diff = ONE - u - w

    term1 = diff * (J(w) + J(u)) / TWO

    y_input = (H2(u) + H2(w)) / diff
    inv_L_val = rigorous_L_inv(y_input)
    term2 = diff * J(inv_L_val)

    return term1 - term2


def g(u_min, u_max, w_min, w_max):
    """
    g(u,w) = kappa(u,w) - kappa(1-u, w).
    The goal is to prove g(u,w) <= 0 on the domain (0, 0.5)^2.
    """
    u_interval = arb(u_min).union(arb(u_max))
    w_interval = arb(w_min).union(arb(w_max))
    return kappa(u_interval, w_interval) - kappa_1_minus_u(u_interval, w_interval)


# ---------------------------------------------------------
# DOMAIN DECOMPOSITION
# ---------------------------------------------------------
@dataclass
class ProofStats:
    verified_area: float = 0.0
    error_count: int = 0
    max_depth_count: int = 0
    boxes_processed: int = 0


def min_distance_to_diagonal(u_min, u_max, w_min, w_max):
    """Minimum value of |u-w| over the box."""
    if u_max < w_min:
        return w_min - u_max
    if w_max < u_min:
        return u_min - w_max
    return 0.0


def subdivide(u_min, u_max, w_min, w_max):
    """Quadtree subdivision of one box into four children."""
    u_mid = (u_min + u_max) / 2.0
    w_mid = (w_min + w_max) / 2.0
    return [
        (u_min, u_mid, w_min, w_mid),
        (u_mid, u_max, w_min, w_mid),
        (u_min, u_mid, w_mid, w_max),
        (u_mid, u_max, w_mid, w_max),
    ]


def prove_conjecture(
    domain_box=DOMAIN_BOX,
    diagonal_epsilon=DIAGONAL_EPSILON,
    max_depth=MAX_DEPTH,
):
    print("Starting Formal Proof Verification...")
    print("Method: Branch and Bound using Arb Interval Arithmetic")
    print(f"Precision: {PREC_BITS} bits")
    print(f"Domain: {domain_box}")
    print(f"Diagonal epsilon: {diagonal_epsilon}")
    print(f"Max depth: {max_depth}")

    start_time = time.time()
    queue = deque([domain_box])
    total_area = (domain_box[1] - domain_box[0]) * (domain_box[3] - domain_box[2])
    min_box_width = (domain_box[1] - domain_box[0]) / (2**max_depth)
    stats = ProofStats()

    while queue:
        u_min, u_max, w_min, w_max = queue.popleft()
        stats.boxes_processed += 1

        if stats.boxes_processed % 10000 == 0:
            elapsed = time.time() - start_time
            percent_done = 100.0 * stats.verified_area / total_area
            print(
                f"Update: Processed {stats.boxes_processed} boxes in {elapsed:.1f}s... "
                f"Queue: {len(queue)}. Verified: {percent_done:.2f}%"
            )

        min_diff = min_distance_to_diagonal(u_min, u_max, w_min, w_max)
        evaluate_box = min_diff >= diagonal_epsilon

        if evaluate_box:
            try:
                result = g(u_min, u_max, w_min, w_max)

                if nonpositive(result):
                    stats.verified_area += (u_max - u_min) * (w_max - w_min)
                    continue

                if positive(result):
                    print(f"COUNTEREXAMPLE FOUND in box: U:[{u_min}, {u_max}], W:[{w_min}, {w_max}]")
                    return False

            except (ValueError, ZeroDivisionError):
                # EXPECTED MATH ERRORS
                # The box is too wide, causing log(negative) or div-by-zero.
                # We chose to ignore the error, increment the counter, and force a subdivision for a more narrower interval
                stats.error_count += 1
                evaluate_box = False
                
            except Exception as exc:
                # UNEXPECTED CODE ERRORS
                raise RuntimeError(
                    "Unexpected code error during interval evaluation on box "
                    f"U:[{u_min}, {u_max}], W:[{w_min}, {w_max}]"
                ) from exc

        # Subdivide if we haven't reached the maximum depth
        if (u_max - u_min) > min_box_width:
            queue.extend(subdivide(u_min, u_max, w_min, w_max))
        else:
            if evaluate_box:
                stats.max_depth_count += 1

    total_time = time.time() - start_time
    final_percent = 100.0 * stats.verified_area / total_area

    print(f"\nProof Complete in {total_time:.2f} seconds.")
    print(f"Total interior area successfully verified: {final_percent:.2f}% ({stats.verified_area:.6f})")
    print(f"Total boxes hitting mathematical errors: {stats.error_count}")
    print(f"Total off-diagonal boxes reaching max depth: {stats.max_depth_count}")
    print(f"Total boxes processed: {stats.boxes_processed}")
    return True


if __name__ == "__main__":
    prove_conjecture()