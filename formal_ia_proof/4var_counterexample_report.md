# Formal Certificate for the Four-Variable Counterexample Candidate

Date: 2026-04-22

## Candidate

The candidate tuple is

```text
(mu, mw, eu, ew) = (6.6e-7, 4.0e-7, 1.1e-5, 9.0e-6).
```

It is tested against the four-variable inequality stated in the repository
README and implemented in `numerical_tests/4var_Inequality.py`.

The inequality requires

```text
LHS - RHS >= 0.
```

Therefore, a certified value `LHS - RHS < 0` proves that the tuple is a
counterexample to that stated inequality.

## Method

The file `formal_ia_proof/certify_4var_counterexample.py` is a dedicated Arb
certificate script for this single tuple.

It uses:

```text
python-flint 0.8.0
Arb ball arithmetic
1536 bits of precision
620 bisection steps for inverse enclosures
```

The script does not use PyTorch, random sampling, grid search, or tolerance
thresholds.

## Inverse Functions

The functions `H2^-1` and `L^-1` are certified by monotone interval bisection.

This uses the analytic facts:

```text
H2'(x) = J(x) = log2((1-x)/x) > 0        on 0 < x < 1/2
```

and

```text
L'(x) =
  (2 J(x) (1 - 2x) + 4 H2(x)) / (1 - 2x)^2 > 0
                                                on 0 < x < 1/2.
```

Thus both inverse branches used by the certificate are single-valued and
monotone on the required domain.

## Certified Domain Checks

The script certifies:

```text
H2(mu) - eu > 0
H2(mw) - ew > 0
```

The computed Arb enclosures have positive lower bounds:

```text
H2(mu) - eu = 3.5026586349963607429471678977836024e-6  +/- 4.19e-462
H2(mw) - ew = 7.8476566624581277113611120408173182e-8  +/- 9.26e-463
```

So the tuple satisfies the stated admissibility constraints.

## Certified Inequality Evaluation

The script certifies:

```text
LHS = 3.1563132157258175938439778884547978e-8  +/- 8.54e-181
RHS = 3.1956325201212080134818614292708769e-8  +/- 6.87e-180
```

and therefore:

```text
LHS - RHS =
  -3.931930439539041963788354081607916e-10  +/- 1.38e-179.
```

The upper endpoint of this Arb ball is strictly negative. Hence:

```text
LHS - RHS < 0.
```

## Conclusion

Subject to the standard trust assumption that Arb/python-flint implements
correct outward-rounded ball arithmetic, the tuple

```text
(6.6e-7, 4.0e-7, 1.1e-5, 9.0e-6)
```

is a certified counterexample to the four-variable inequality as stated in the
repository README and as encoded by the certificate script.


