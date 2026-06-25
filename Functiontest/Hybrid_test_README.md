# Hybrid System MPC — Log-Transform Saturation Nesting

## 1. Overview

Constrained MPC with three cost levels in **strict lexicographic priority**:

| Priority | Level | Meaning | Typical raw magnitude |
|----------|-------|---------|----------------------|
| 1 (highest) | L1 | Static obstacle violation (walls, pillars, bounds) | 0–200 |
| 2 | L2 | Dynamic chance constraint (pedestrian MC quantile) | 0–30 |
| 3 (lowest) | L3 | Running cost (tracking + control effort) | 0–1e8 (甚至负值) |

**Requirement**: ANY L1 violation must cost more than ANY L2 violation,
which must cost more than ANY L3 value. The MMOG-IGO solver uses
`argsort` ranking — it only needs to *compare* costs, not differentiate
them. But all values must be **distinguishable in float32** (≈7 digits).

**Key design goal**: The solver should work as a **generic solver** —
the cost function $f$ may be a black box with unknown magnitude,
potentially as large as $10^8$, and may even take **negative values**
(when the cost includes reward terms). The scheme must handle all of
this without per-problem tuning.

---

## 2. Core Building Blocks

### 2.1 Saturation Function

$$
\boxed{\sigma_k(x) = \frac{kx}{\sqrt{1 + (kx)^2}}}
$$

- **Odd function**: $\sigma_k(-x) = -\sigma_k(x)$ — preserves sign symmetry.
- **Output**: strictly in $(-1, 1)$, **independent of $k$** in the limit.
- **$k$ controls the knee**: $\sigma_k(1/k) = 1/\sqrt{2} \approx 0.707$.
  - $x \in [0, 1/k]$: approximately linear, $\sigma_k(x) \approx kx$
  - $x \gg 1/k$: saturated, $\sigma_k(x) \to 1$
- **Variable $k$ is the key knob** for tuning sensitivity range.

```
σ_k(x)
1.0 ┤                                   ╭─────
    ┤                             ╭─────╯
    ┤                       ╭─────╯          k=1.0 (knee at x=1)
    ┤                  ╭────╯
    ┤             ╭────╯                     k=0.5 (knee at x=2)
    ┤        ╭────╯
    ┤   ╭────╯                               k=0.2 (knee at x=5)
    ┤╭──╯
    ┤╯                                       k=0.05 (knee at x=20)
    ┼─────────────────────────────────────
    0    5    10   15   20
```

### 2.2 Log Transform

$$
\boxed{\mathcal{T}(x) = \text{sign}(x) \cdot \log(1 + |x|)}
$$

- **Odd function**: $\mathcal{T}(-x) = -\mathcal{T}(x)$ — handles negative costs
  with exactly the same precision as positive costs.
- $x \approx 0$: $\mathcal{T}(x) \approx x$ (linear near zero — preserves
  small-value sensitivity).
- Large $|x|$: $\mathcal{T}(x) \approx \text{sign}(x) \cdot \log|x|$ (logarithmic
  compression — prevents saturation).
- Replaces the need to estimate characteristic scales $S_1, S_2, S_3$.

### 2.3 Why Log + Variable $k$ Instead of Auto-Scaling?

The original approach required estimating $S_3 \approx (H+15)\cdot\|s-t\|^2$
to normalize $f$ into $[0, 1]$. This fails when:

1. $f$ is a **black box** (e.g., learned cost, arbitrary user function).
2. $f$ can reach **$10^8$** or more — $S_3$ estimation becomes unreliable.
3. $f$ can be **negative** — the $f/S_3$ normalization doesn't handle signs
   naturally.

The log-transform approach solves all three:

- $\mathcal{T}(f)$ compresses any $[0, 10^8]$ range down to $[0, \sim 18]$
  automatically, without knowing $f_{\max}$.
- A smaller $k$ on the innermost $\sigma$ shifts the knee to match the
  compressed range.
- Sign symmetry handles negative costs cleanly.

---

## 3. Three-Level Nesting Structure

$$
\mathcal{L}(z) = \sigma_1\!\left(
\begin{cases}
\mathcal{T}(g_1) + \delta_1 & \text{if } g_1 > 0 \quad \text{(L1: static violation)} \\[8pt]
\sigma_1\!\left(
\begin{cases}
\mathcal{T}(g_2) + \delta_2 & \text{if } g_2 > 0 \quad \text{(L2: dynamic violation)} \\[8pt]
\sigma_{k_3}\!\big(\mathcal{T}(f)\big) & \text{otherwise} \quad \text{(L3: running cost)}
\end{cases}
\right)
\end{cases}
\right)
$$

where:

- $g_1(z)$ = total static violation over horizon $H$
- $g_2(z)$ = total dynamic violation over horizon $H$
- $f(z)$ = total running cost over horizon $H$ (may be negative!)
- $\mathcal{T}(x) = \text{sign}(x) \cdot \log(1 + |x|)$ — log transform
- $\delta_1 = 1.5$, $\delta_2 = 3.0$ — fixed offsets (ensure strict hierarchy)
- $k_3 = 0.1$ — reduced knee for the innermost $\sigma$ (handles large $f$ range)
- Outer $\sigma$ use $k = 1.0$

### Why $k_3 = 0.1$?

The innermost $\sigma$ is closest to the raw cost. It must handle the widest
dynamic range. With $\mathcal{T}(f_{\max}) \approx \log(1 + 10^8) \approx 18.4$:

| $k_3$ | Knee at input | Knee at $f$ | $f \in [10^7, 10^8]$ distinguishable? |
|-------|---------------|-------------|--------------------------------------|
| 1.0   | 1             | $\approx 1.7$ | ✗ (~1,400 values — marginal) |
| 0.1   | 10            | $\approx 2.2\times 10^4$ | ✓ (~123,000 values) |
| 0.05  | 20            | $\approx 4.8\times 10^8$ | ✓✓ (~330,000 values) |

**Chosen: $k_3 = 0.1$** — good distinguishability across the full range
$f \in [0, 10^8]$ while keeping reasonable sensitivity at small $f$.

---

## 4. Log-Transform Properties

### 4.1 Dynamic Range Compression

For $f \in [0, 10^8]$:

| $f$ | $\mathcal{T}(f)$ | $\sigma_{0.1}(\mathcal{T}(f))$ | After triple $\sigma$ |
|-----|------------------|-------------------------------|----------------------|
| 0 | 0 | 0 | 0 |
| 0.01 | 0.010 | 0.001 | 0.001 |
| 1 | 0.693 | 0.069 | 0.069 |
| 100 | 4.615 | 0.419 | 0.360 |
| $10^4$ | 9.210 | 0.678 | 0.489 |
| $10^6$ | 13.82 | 0.811 | 0.533 |
| $10^8$ | 18.42 | 0.879 | 0.551 |
| $\to \infty$ | $\to \infty$ | $\to 1$ | $\to 0.577$ |

The log transform compresses 8 orders of magnitude into a manageable
$[0, 18]$ range, and $k_3 = 0.1$ ensures the innermost $\sigma$ operates
mostly below its knee ($1/k_3 = 10$).

### 4.2 Sign Symmetry for Negative Costs

Since both $\mathcal{T}$ and $\sigma_k$ are odd functions, the entire
nesting is odd:

$$
\mathcal{L}(-f) = -\mathcal{L}(f)
$$

This means negative costs (reward-like terms, $f < 0$) are handled with
**exactly the same precision** as positive costs. The ranking of two
negative costs $f_a < f_b < 0$ is correctly preserved.

| $f$ | $\mathcal{L}(f)$ | | $-f$ | $\mathcal{L}(-f)$ |
|-----|------------------|---|------|-------------------|
| 0 | 0 | | 0 | 0 |
| 100 | +0.360 | | -100 | -0.360 |
| $10^4$ | +0.489 | | $-10^4$ | -0.489 |
| $10^8$ | +0.551 | | $-10^8$ | -0.551 |

---

## 5. Hierarchy Proof

### 5.1 Offset Selection

With the log transform and $k_3 = 0.1$, the cascade depth determines
maximum possible output:

$$
\begin{aligned}
\sigma_{0.1}(\mathcal{T}(f)) &\in [0,\; 1) \quad \text{for } f \ge 0 \\
\sigma_1(\sigma_{0.1}(\mathcal{T}(f))) &\in [0,\; \sigma_1(1)) = [0,\; 0.707) \\
\sigma_1(\sigma_1(\sigma_{0.1}(\mathcal{T}(f)))) &\in [0,\; \sigma_1(0.707)) = [0,\; 0.577)
\end{aligned}
$$

For strict L1 > L2 > L3 we need:

$$
\begin{aligned}
\text{L1 min} = \sigma_1(\delta_1) &> \max(\text{L2}) = \sigma_1(\sigma_1(\mathcal{T}(g_{2,\max}) + \delta_2)) \\
\text{L2 min} = \sigma_1(\sigma_1(\delta_2)) &> \max(\text{L3}) = \sigma_1(\sigma_1(1)) = 0.577
\end{aligned}
$$

With $\mathcal{T}(g_{2,\max}) = \log(1 + 26) \approx 3.30$:

| $\delta_1$ | $\sigma_1(\delta_1)$ | Meets L1 > L2_max? |
|-----------|---------------------|---------------------|
| 1.4 | 0.814 | ✓ (if L2_max < 0.814) |
| 1.5 | 0.832 | ✓ (more margin) |

| $\delta_2$ | $\sigma_1(\sigma_1(\delta_2))$ | Meets L2 > L3 (0.577)? |
|-----------|-------------------------------|------------------------|
| 2.5 | 0.651 | ✓ (margin 0.074) |
| 3.0 | 0.688 | ✓ (margin 0.111) |

**Chosen**: $\delta_1 = 1.5$, $\delta_2 = 3.0$.

### 5.2 Output Ranges

**Case A — Static violation** ($g_1 > 0$):

$$
\mathcal{T}(g_1) + 1.5 \geq 1.5 \;\Rightarrow\; \mathcal{L} \in [\sigma_1(1.5),\; 1) = \mathbf{[0.832,\; 1.0)}
$$

**Case B — Dynamic violation only** ($g_1 = 0$, $g_2 > 0$):

$$
\begin{aligned}
\mathcal{T}(g_2) + 3.0 &\geq 3.0 \\
\sigma_1(\geq 3.0) &\in [0.949,\; 1.0) \quad \text{(L2 output)} \\
\sigma_1([0.949, 1.0)) &\in [\sigma_1(0.949),\; \sigma_1(1.0)) \\
&= \mathbf{[0.688,\; 0.707)}
\end{aligned}
$$

**Case C — No violation** ($g_1 = 0$, $g_2 = 0$):

$$
\begin{aligned}
\mathcal{T}(f) &\in (-\infty,\; +\infty) \\
\sigma_{0.1}(\mathcal{T}(f)) &\in (-1,\; 1) \quad \text{(L3 innermost)} \\
\sigma_1(\sigma_{0.1}(\mathcal{T}(f))) &\in (-0.707,\; 0.707) \\
\sigma_1(\sigma_1(\sigma_{0.1}(\mathcal{T}(f)))) &\in (-0.577,\; 0.577)
\end{aligned}
$$

For $f \ge 0$: $\mathbf{[0,\; 0.577)}$; for $f < 0$: $\mathbf{(-0.577,\; 0]}$.

### 5.3 Summary

```
┌──────────────────────┬──────────────────────┐
│ Level                │ Cost range           │
├──────────────────────┼──────────────────────┤
│ L1 (static viol)     │ [0.832, 1.000)       │
│ L2 (dyn viol only)   │ [0.688, 0.707)       │
│ L3 (f ≥ 0, no viol)  │ [0.000, 0.577)       │
│ L3 (f < 0, no viol)  │ (-0.577, 0.000]      │
└──────────────────────┴──────────────────────┘
```

$$
\boxed{\min(\text{L1}) = 0.832 \;>\; \max(\text{L2}) = 0.707 \;\gg\; \max(\text{L3}_{f\ge 0}) = 0.577 \quad \checkmark}
$$

Margins: L1–L2 ≈ 0.125, L2–L3 ≈ 0.111 — both comfortably above float32
resolution ($\sim 6\times 10^{-8}$ at magnitude 0.5).

---

## 6. Float32 Distinguishability

### 6.1 With Log Transform + $k_3 = 0.1$

Each level spans a wide output range, giving millions of distinguishable
values:

| $f$ interval | Output width | Distinguishable values |
|-------------|-------------|----------------------|
| $[10^{-2}, 1]$ | 0.068 | ~1,130,000 |
| $[1, 10^3]$ | 0.374 | ~6,240,000 |
| $[10^3, 10^5]$ | 0.073 | ~1,220,000 |
| $[10^5, 10^7]$ | 0.027 | ~457,000 |
| $[10^7, 10^8]$ | 0.007 | ~123,000 |

Even in the most demanding interval ($f \in [10^7, 10^8]$), we have
**~123,000 distinguishable values** — more than 1000× the ~80 samples
ranked per solver iteration.

### 6.2 Why the Original Auto-Scaling Failed for Large $f$

If $f$ reaches $10^8$ but $S_3$ is estimated for a smaller range:

Without normalization (raw $f \approx 10^8$ enters triple saturation):

$$
\sigma_1(\sigma_1(\sigma_1(10^8))) = 0.57735
$$
$$
\sigma_1(\sigma_1(\sigma_1(10^7))) = 0.57735
$$

Difference: $\sim 10^{-9}$. **Zero distinguishable values** → `argsort` is
random → solver degenerates → NaN.

The log transform fixes this because $\mathcal{T}(10^8) \approx 18.4$ vs
$\mathcal{T}(10^7) \approx 16.1$ — a difference of 2.3 in the input to
the first $\sigma$, which is easily resolved.

### 6.3 Negative Cost Precision

Since the entire transform chain is odd, negative costs have identical
precision to their positive counterparts:

| $f$ interval | Distinguishable values |
|-------------|----------------------|
| $[-10^8, -10^6]$ | ~123,000 |
| $[-10^6, -10^4]$ | ~457,000 |
| $[-1, -0.01]$ | ~1,130,000 |

---

## 7. Solver Numerical Configuration (MPCsolverM22)

### 7.1 Precision Matrix Eigenvalue Bounds

```python
MIN_EIG = 1e-3   # minimum eigenvalue of precision matrix S
MAX_EIG = 1e3    # maximum eigenvalue of precision matrix S
```

The precision matrix $S$ has eigenvalues clamped to $[10^{-3}, 10^3]$.
Condition number $\kappa \leq 10^6$, ensuring `eigh` and `solve` retain
$\sim 1$ digit of accuracy in float32 ($\varepsilon \approx 1.2 \times 10^{-7}$,
error $\approx \kappa \cdot \varepsilon \approx 0.1$).

Without clamping, $\kappa \to 10^{12}$, giving complete loss of precision.

### 7.2 IGO Weight Clipping

```python
a_i = exp(clip(log_pdf_k - log_mog, -20, 20))
```

The IGO weight $a_i = \exp(\log p_k - \log p_{\text{MoG}})$ multiplies
the precision matrix gradient $S \cdot \text{outer}(d,d) \cdot S$.

- With clip 70: $a_i \leq e^{70} \approx 2.5 \times 10^{30}$
  → $a_i \cdot S \cdot d \cdot d^\top \cdot S \approx 3 \times 10^{36}$
  → **near float32 max** ($3.4 \times 10^{38}$) → **overflow → NaN**

- With clip 20: $a_i \leq e^{20} \approx 4.8 \times 10^8$
  → product $\approx 6 \times 10^{14}$ → **well within float32 range**

---

## 8. Complete Cost Function Code

```python
@jit
def sigma_k(x, k=1.0):
    """σ_k(x) = kx / √(1 + (kx)²), odd function, output ∈ (-1, 1)"""
    kx = k * x
    return kx / jnp.sqrt(1.0 + kx**2)

@jit
def log_transform(x):
    """sign(x) * log(1 + |x|) — odd, compresses dynamic range"""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

@jit
def mpc_cost_fn(z_flat, ctx):
    (init_state, target, dt_val, _,
     ped1_traj, ped1_r, ped1_sigma, mc_key) = ctx
    H = z_flat.shape[0] // 2

    # ... rollout, compute f_total, viol_dynamic, viol_static ...

    # ─── Log-transform nesting (no per-problem scale parameters) ───
    # δ₁=1.5, δ₂=3.0, k_inner=0.1 — universal constants
    res = sigma_k(                                   # L1: k=1.0 (default)
        jnp.where(viol_static > 0.0,
                  log_transform(viol_static) + 1.5,  # δ₁ = 1.5
        sigma_k(                                      # L2: k=1.0 (default)
            jnp.where(viol_dynamic > 0.0,
                      log_transform(viol_dynamic) + 3.0,  # δ₂ = 3.0
            sigma_k(log_transform(f_total), k=0.1)   # L3: k=0.1 for wide range
        ))
    )
    return res
```

### Design Rationale

| Component | Choice | Why |
|-----------|--------|-----|
| `log_transform` | $\text{sign}(x)\log(1+|x|)$ | Compresses $[0,10^8]$ to $[0,18]$, handles negatives |
| L3 $\sigma$ | $k=0.1$ | Knee at $f\approx 22000$, good sensitivity to $10^8$ |
| L2 $\sigma$ | $k=1.0$ | $g_2$ range is moderate ($\log(1+26)\approx 3.3$) |
| L1 $\sigma$ | $k=1.0$ | $g_1$ range is moderate ($\log(1+174)\approx 5.2$) |
| $\delta_1$ | 1.5 | Ensures L1 min (0.832) > L2 max (0.707) |
| $\delta_2$ | 3.0 | Ensures L2 min (0.688) > L3 max (0.577) |

---

## 9. Adapting to a New Problem

### The parameters are universal

- $\mathcal{T}(x) = \text{sign}(x)\log(1+|x|)$ — always
- $\delta_1 = 1.5$, $\delta_2 = 3.0$ — always
- $k_3 = 0.1$ — works for $f$ up to $\sim 10^8$
- Solver: `MIN_EIG=1e-3`, `MAX_EIG=1e3`, IGO clip=20

### Only one thing to check: $k_3$

If $f$ can be much larger than $10^8$, reduce $k_3$:

| $f_{\max}$ approx | Recommended $k_3$ | Knee at $f$ |
|-------------------|-------------------|-------------|
| $\sim 10^4$ | 0.5 | $\sim 6$ |
| $\sim 10^6$ | 0.2 | $\sim 150$ |
| $\sim 10^8$ | **0.1** | $\sim 2.2\times 10^4$ |
| $\sim 10^{12}$ | 0.05 | $\sim 4.8\times 10^8$ |
| $\sim 10^{20}$ | 0.02 | $\sim 5.2\times 10^{21}$ |

Rule of thumb: pick $k_3$ such that $\log(1 + f_{\max}) \lesssim 2/k_3$.

**No per-problem computation of $S_1, S_2, S_3$ is needed.** The log
transform handles the dynamic range automatically.

### If $f$ is a black box

- **No estimation of $f_{\max}$ is required.** The log transform
  compresses any range; $k_3 = 0.1$ is a safe default for most problems.
- If you observe that the solver can't distinguish different $f$ values
  (outputs are all near 0.55), reduce $k_3$.
- If you observe that small $f$ differences are lost (outputs are all
  near 0), increase $k_3$.

---

## Appendix: Smooth Version (No `jnp.where`)

The `jnp.where`-based structure can be smoothed into a single continuous
expression:

$$
\mathcal{L} = \sigma_1\!\left(
\mathcal{T}(v_1) + \delta_1\sigma_1(\beta_1 v_1) +
\sigma_1\!\left(
\mathcal{T}(v_2) + \delta_2\sigma_1(\beta_2 v_2) +
\sigma_{k_3}\!\left(\mathcal{T}(f)\right)
\right)\right)
$$

where $\beta_1, \beta_2$ control hardness ($\beta \to \infty$ approximates
`jnp.where`), $\delta_1, \delta_2$ control jump magnitude.
No `jnp.where` at all — differentiable everywhere, friendly to
gradient-based solvers.

```python
@jit
def mpc_cost_fn_smooth(z_flat, ctx):
    # ... rollout ...

    beta1, beta2 = 100.0, 100.0  # hardness (larger → sharper transition)

    return sigma_k(
        log_transform(viol_static) + 1.5 * sigma_k(beta1 * viol_static)
        + sigma_k(
            log_transform(viol_dynamic) + 3.0 * sigma_k(beta2 * viol_dynamic)
            + sigma_k(log_transform(f_total), k=0.1)
        )
    )
```
