# LPAP — pseudocode draft (plain / markdown)

Working draft for the LaTeX algorithm frame in the README.
Matches `lpap.ops.lpap_torch` (real signs). The Manim film flips chalk signs
for pedagogy; do **not** copy those here.

## Choices (locked for now)

- Symbols over full names (`V`, `Δ`, `a`, `d`, …).
- Lane remap `(i − k) mod C` instead of an explicit `roll` (more concise;
  same probing process).
- Python-like `where m:` for the masked swap / write.
- Conceptual return is $(a,d)$; residual work / permutation live in notes.

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
LPAP(x ∈ ℝᴺ; C, K) → (a ∈ ℝᶜ, d ∈ ℤᶜ)
────────────────────────────────────────
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

return a, d
```

Only the `for k` loop is serial. Over buckets $i$, the map $i \mapsto \ell_i$
is a bijection, so the body is one vectorized step.

---

## Notes (on the figure)

1. **Residual work.** The reference implementation also returns
   $\mathrm{flatten}(V)$, the residual work array after in-place swap-backs. It
   is scratch state, not part of the pooled representation $(a,d)$; $\Delta$
   likewise.

2. **Grouped permutation.** In the full stack, a fixed grouped permutation is
   applied to $x$ before this procedure and inverted afterward. Contiguous
   source groups are scattered across bucket lanes so structure in the input
   cannot concentrate large amplitudes in a few lanes. With large enough $K$,
   the $C$ largest-magnitude values then have a fair chance to enter the table.

Keep the “preference for random algorithms” motivation out of user-facing text;
the figure states the operational reason (decorrelation / coverage under bounded
$K$).

---

## TeX → README image (Pixi)

```bash
pixi run tex-lpap-algo       # PDF + doc/assets/lpap-algorithm.png
pixi run tex-lpap-algo-pdf   # PDF only under doc/tex/build/
```

Source: [`doc/tex/lpap_algorithm.tex`](tex/lpap_algorithm.tex).
