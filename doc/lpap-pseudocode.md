# LPAP — pseudocode draft (plain / markdown)

Working draft toward a LaTeX algorithm frame for the README.
Matches `lpap.ops.lpap_torch` (real signs). The Manim film flips chalk signs
for pedagogy; do **not** copy those here.

## Choices (locked for now)

- Symbols over full names (`V`, `Δ`, `a`, `d`, …).
- Lane remap `(i − k) mod C` instead of an explicit `roll` (more concise;
  same probing process).
- Python-like `where m:` for the masked swap / write.

## Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $N$ | — | number of values ($N$ multiple of $C$) |
| $C$ | — | bucket count |
| $P = N/C$ | — | probes per lane |
| $K$ | — | max rolls (`k_max`) |
| $V$ | $C \times P$ | work view of the input (in-place) |
| $\Delta$ | $C \times P$ | parking tickets (`dibs_diff`) |
| $a$ | $C$ | bucket amplitudes |
| $d$ | $C$ | distances-in-buckets (`dibs`) |
| $k$ | scalar | roll index, $k = 0,\ldots,K-1$ |
| $\ell_i$ | — | source lane $(i - k) \bmod C$ |
| $m$ | $C$ | amplitude gate (boolean / \{0,1\}) |

Amplitude uses $|\cdot|$; signed values stay in $a$ and $V$.
Batch axis omitted (`vmap`).

---

## Algorithm

```text
LPAP(x ∈ ℝᴺ; C, K) → (a ∈ ℝᶜ, d ∈ ℤᶜ, x' ∈ ℝᴺ)
──────────────────────────────────────────────────
require  N mod C = 0
P ← N / C
V ← reshape(x, C × P)
Δ ← 0_{C×P}
a ← 0_C
d ← 0_C

for k = 0 … K−1 do
    ℓ_i ← (i − k) mod C                 # ∀ i ∈ {0…C−1}
    j_i ← argmax_p |V[ℓ_i, p]|
    c_i ← V[ℓ_i, j_i]
    d̂_i ← Δ[ℓ_i, j_i] + k
    m_i ← (|c_i| ≥ |a_i|)

    where m:
        V[ℓ, j] ← a
        Δ[ℓ, j] ← d − k
        a       ← c
        d       ← d̂

return a, d, flatten(V)
```

Only the `for k` loop is serial. Over buckets $i$, the map $i \mapsto \ell_i$
is a bijection, so the body is one vectorized step.

---

## Notes for LaTeX later

1. Title: Linear Probing Amplitude Pooling (LPAP).
2. Typeset `where m:` like a Python block under the gate line.
3. Optional footnote: film chalk uses $k - d$; code uses $d - k$.

## TeX → README image (Pixi)

No system TeX needed. Project deps: `tectonic` + `poppler` (via Pixi).

```bash
pixi run tex-lpap-algo       # PDF + doc/assets/lpap-algorithm.png
pixi run tex-lpap-algo-pdf   # PDF only under doc/tex/build/
```

Source: [`doc/tex/lpap_algorithm.tex`](tex/lpap_algorithm.tex).
Build intermediates are gitignored; commit the PNG when you want it in the README.
