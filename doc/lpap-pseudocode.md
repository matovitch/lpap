# LPAP — pseudocode draft (plain / markdown)

Working draft for the LaTeX algorithm frame in the README.
Matches `lpap.ops.lpap_tokens_torch` and `lpap.triton_ops.lpap_tokens_triton`
(real signs, column-major fold, explicit occupancy flags). The Manim film flips
chalk signs for pedagogy; do **not** copy those here.

## Choices

- Symbols over full names (`V`, `U`, `Δ`, `a`, `d`, `v`, …).
- Column-major fold `V[b, p] ← x[p·C + b]`.
- Lane remap `(i − k) mod C` instead of an explicit `roll`.
- Explicit occupancy masks `U` (`work_valid`) and `v` (`bucket_valid`) so empty
  buckets accept real zeros without swapping phantom zeros back into `V`.
- Python-like `where m:` for the masked swap / write.
- Conceptual return is $(a, d)$; residual work and permutation live in notes.

## Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $N$ | — | number of values ($N > 0$, $N$ multiple of $C$) |
| $C$ | — | bucket count |
| $P = N/C$ | — | probes per lane |
| $K$ | — | max rolls (`k_max`, $1 \le K \le C$) |
| $V$ | $C \times P$ | working copy of column-major folded input |
| $U$ | $C \times P$ | work-slot occupancy mask (`work_valid`) |
| $\Delta$ | $C \times P$ | parking tickets (`dibs_diff`) |
| $a$ | $C$ | bucket amplitudes |
| $d$ | $C$ | distances-in-buckets (`dibs`) |
| $v$ | $C$ | bucket occupancy mask (`bucket_valid`) |
| $k$ | scalar | roll index, $k = 0,\ldots,K-1$ |
| $\ell_i$ | — | source lane $(i - k) \bmod C$ |
| $m$ | $C$ | acceptance gate (boolean / $\{0,1\}$) |

Amplitude uses $|\cdot|$; signed values stay in $a$ and $V$.
Batch axis omitted (`vmap`).

---

## Algorithm

```text
LPAP(x ∈ ℝᴺ; C, K) → (a ∈ ℝᶜ, d ∈ ℤᶜ)
────────────────────────────────────────
require  N > 0,  N mod C = 0,  1 ≤ K ≤ C
P ← N / C
V[b, p] ← x[p·C + b]                    # column-major copy
U ← 1_{C×P}
Δ ← 0_{C×P}
a ← 0_C
d ← 0_C
v ← 0_C

for k = 0 … K−1 do
    ℓ_i ← (i − k) mod C                 # ∀ i ∈ {0…C−1}
    s_{i,p} ← |V[ℓ_i, p]|  if U[ℓ_i, p] ∧ finite(V[ℓ_i, p])  else −1
    j_i ← argmax_p s_{i,p}
    c_i ← V[ℓ_i, j_i]
    d̂_i ← Δ[ℓ_i, j_i] + k
    m_i ← (s_{i,j_i} ≥ 0) ∧ (¬v_i ∨ |c_i| ≥ |a_i|)

    where m:
        V[ℓ, j] ← a  if v  else 0
        U[ℓ, j] ← v
        Δ[ℓ, j] ← d − k  if v  else 0
        a       ← c
        d       ← d̂
        v       ← 1

return a, d
```

Only the `for k` loop is serial. Over buckets $i$, the map $i \mapsto \ell_i$
is a bijection, so the body is one vectorized step.

---

## Notes (on the figure)

1. **Residual work & probe tracking.** The implementation copies $x$ into $V$
   (input $x$ is unmodified) and also returns the unfolded residual work array
   (zeroed wherever $U = 0$) and per-bucket probe indices for local surrogate
   targets. Residual work, $U$, $v$, and $\Delta$ are internal state, not part
   of the pooled representation $(a, d)$.

2. **Random permutation.** In the full stack, a fixed seeded random
   permutation is applied to $x$ before column-major folding and inverted
   afterward, so structured locality in the input does not align with the
   folded bucket lanes.

---

## TeX → README image (Pixi)

```bash
pixi run tex-lpap-algo       # PDF + doc/assets/lpap-algorithm.png
pixi run tex-lpap-algo-pdf   # PDF only under doc/tex/build/
```

Source: [`doc/tex/lpap_algorithm.tex`](tex/lpap_algorithm.tex).
