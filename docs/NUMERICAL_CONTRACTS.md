# Numerical contracts

These identities are the oracles for later tests. An implementation that disagrees with this file is wrong even if it is internally consistent. Expected test values are derived here or from the Python `datetime` Gregorian calendar, not by calling the function under test.

## Sphere

Mean radius `R = 6_371_000 m`.

Regular cell area, with latitude bounds in radians and longitude width in radians:

```
area = R^2 * (sin(lat_north) - sin(lat_south)) * delta_lon
```

`delta_lon` is positive and at most `2 pi`. A cell whose longitude bounds run backwards across the branch cut has width `(lon_east + 2 pi - lon_west)` after both edges are unwrapped into a continuous interval.

The area integral of a constant field `c` over a set of cells is `c` times the sum of the cell areas. The area-weighted mean of a complete constant field is `c`. The unweighted mean of a constant field is also `c`. On a field that varies with latitude, the two means differ, and that difference is the unweighted error.

## Missing values

IEEE NaN is the only missing marker. A sum skips non-finite inputs and also skips their weights. If the retained weight is zero, the mean is NaN. No function in the core replaces NaN with zero.

## Annualization

For monthly means `m_k` with month lengths `d_k` on the dataset calendar:

```
annual = sum(m_k d_k) / sum(d_k)
```

taken only over finite months. Coverage is `sum(d_k) / days_in_year`. On the 360-day calendar every `d_k` is 30. On noleap, February is 28. On proleptic Gregorian, February is 29 only in leap years, and year 1900 is not a leap year while year 2000 is.

## Carbon impulse response

Joos et al. (2013) multi-model mean, declared as the default airborne fraction after a pulse of carbon:

```
G(t) = a0 + a1 exp(-t/tau1) + a2 exp(-t/tau2) + a3 exp(-t/tau3)
a = (0.2173, 0.2240, 0.2824, 0.2763)
tau = (394.4, 36.54, 4.304) years
```

`a0 + a1 + a2 + a3 = 1`. The equivalent pools are

```
dC0/dt = a0 E(t)
dCi/dt = ai E(t) - Ci / tau_i    for i = 1, 2, 3
C = C0 + C1 + C2 + C3
```

A unit pulse adds `ai` to pool `i` and nothing thereafter. As `t` goes to infinity, `C` goes to `a0`.

Mass conversion: `1 ppm CO2 = 2.124 GtC` unless overridden. `1 GtC = (44/12) GtCO2`.

## CO2 forcing

```
F = (F_2x / ln(2)) * ln(C / C_ref)
F_2x = 3.93 W m-2
```

At `C = 2 C_ref`, `F = 3.93`. The Myhre form `5.35 ln(C/C_ref)` equals `5.35 ln(2)` at a doubling, about 3.708 W m-2, and is a different convention.

## Two-reservoir energy balance

```
Cs dTs/dt = F - lambda Ts - gamma (Ts - Td)
Cd dTd/dt = gamma (Ts - Td)
```

Matrix form `dx/dt = A x + b F` with `x = (Ts, Td)`:

```
A = [ -(lambda + gamma)/Cs ,  gamma/Cs ]
    [  gamma/Cd             , -gamma/Cd ]
b = [ 1/Cs, 0 ]
```

Trace of `A` is negative and the determinant is `lambda gamma / (Cs Cd)` when all four coefficients are positive. Both eigenvalues then have negative real part. Equilibrium for constant `F` is `Ts = Td = F / lambda` when `lambda != 0`.

One-box limit `gamma = 0`, starting from zero, constant `F`:

```
Ts(t) = (F / lambda) (1 - exp(-lambda t / Cs))
Td(t) = Td(0)
```

Pure integrator limit `lambda = 0`, `gamma = 0`:

```
Ts(t) = Ts(0) + F t / Cs
```

Energy identity, for any trajectory of this linear system:

```
d/dt (Cs Ts + Cd Td) = F - lambda Ts
```

The exchange `gamma (Ts - Td)` cancels between the two layers.

Step response from a zero state:

```
x(t) = (I - exp(A t)) x_eq
x_eq = (F/lambda, F/lambda)
```

`exp(A t)` is computed from the eigen-decomposition of `A`. A test oracle may use a truncated Taylor series with scaling and squaring. The two implementations must agree to a stated tolerance. They must not call each other.

Forward Euler is stable for a step `dt` only when `dt < 2 / |s_fast|` for the eigenvalue `s_fast` of largest magnitude, and only while eigenvalues are real and negative. The solver rejects a larger explicit step rather than returning a diverged state.

## Heat-content and flux

With Earth area `4 pi R^2` and a Julian year of `365.25 * 86400` seconds:

```
ZJ per year = flux_W_m2 * area * year_seconds / 1e21
```

The conversion is an identity of those constants. It does not change a stock of zettajoules into a flux. The stock must be differenced in time first.

## Scores

Root mean square error is the square root of the mean squared difference on finite paired samples. If a pair contains a NaN, the pair is excluded and the retained count is reported. An empty pair set is an error, not a zero score.

Continuous ranked probability score for an ensemble of exchangeable members uses the empirical ensemble at each time and averages over time. Members of one model are exchangeable with each other only inside that model. They are not pooled with another model family into one exchangeable ensemble unless the experiment plan says so.
