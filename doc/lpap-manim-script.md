# LPAP Manim script — detailed storyboard

Status: Act 0–7 done. Use `pixi run manim-full` for the complete film.

**Watch the whole film once** (does not re-run per-act chains):

```bash
pixi run manim-full          # 480p15 → animations/media/.../480p15/LpapFullFilm.mp4
pixi run manim-full-hq       # 1080p60 → animations/media/.../1080p60/LpapFullFilm.mp4
pixi run manim-full-play     # replay last low-quality render
pixi run manim-full-play-hq  # replay last 1080p60 render
pixi run manim-full-preview  # render low-quality + ffplay
pixi run manim-readme        # copy HQ MP4 + build GIF → doc/assets/ (for README)
```

Storyboard acts below are beats inside `LpapFullFilm` (helpers in
`animations/lpap_operator.py`), not separate Pixi scenes.

Target example (`N=12`, `C=4`, `P=3`, `k_max=3`).

**Pedagogy vs real code:** on-screen panel is labeled **`dib_diffs`** (code still
uses `dibs_diff`). Chalk uses `dib_diffs = k − old_dib` when parking an
outgoing value. Re-entry **reads** the candidate’s twin into `dib`
(`dib = k − twin`), then **overwrites** that twin with the outgoing parking
ticket (same write rule). Signs may be flipped vs the implementation.

## Stage layout (horizontal screen)

Orient the work grid as **4 rows × 3 columns**:

- **Rows** = bucket lanes (the dimension that **rolls**)
- **Columns** = probe positions `p0 … p2` (left → right)
- Projection metaphor: within each row, `max |x|` **slides right** into the amplitude bucket for that row

Three main groups, left → right:

1. **Work array** — `4×3` (larger dim vertical: 4 bucket rows × 3 probes).
2. **Amplitude + dibs** — `4×2` center: left column = amplitudes/buckets, right column = dibs (row-aligned with the work rows).
3. **`dib_diffs`** — `4×3` twin of the work array.

```text
                              k=0

   work (4×3)          amp | dib      dib_diffs (4×3)
   p0   p1   p2
  ┌────┬────┬────┐    ┌────┬────┐    ┌────┬────┬────┐
  │ -1 │  6 │  3 │ →  │  6 │  0 │    │  0 │  0 │  0 │
  ├────┼────┼────┤    ├────┼────┤    ├────┼────┼────┤
  │  2 │  5 │  4 │ →  │  5 │  0 │    │  0 │  0 │  0 │
  ├────┼────┼────┤    ├────┼────┤    ├────┼────┼────┤
  │ -8 │  7 │ -5 │ →  │ -8 │  0 │    │  0 │  0 │  0 │
  ├────┼────┼────┤    ├────┼────┤    ├────┼────┼────┤
  │  9 │ -6 │ -4 │ →  │  9 │  0 │    │  0 │  0 │  0 │
  └────┴────┴────┘    └────┴────┘    └────┴────┴────┘
```

Reading left→right along a row: probes → selected amplitude → its dib → (and on the far right) the `dib_diffs` working row.

### Vocabulary on screen

- `N=12 values`, `C=4 buckets`, `K=3 iterations` (do not say `P=N/C` on screen for now).
- Probe count `P=3` is implicit in the grid width.
- Do **not** discuss seeded grouped permutation.

### Notes

- `dib_diffs` has the **same 4×3 shape** as the work grid; always animate it in sync (same rolls, same cell write timing as value swap-backs).
- Center is a **4×2**: amplitudes on the left, dibs on the right — so amplitudes sit next to the work array and dibs sit next to `dib_diffs`.
- Panel tags: **input** (left work grid), **amp** / **dib** (center columns), **dib_diffs** (right). Alt for left: **values** (matches the LPAP API name).
- Act 0 strip and rolling `k` share the same vertical band: centered between the
  `N`/`C`/`K` annotation and the column headers (annotation stays put).
- Stage sits low (small bottom margin) so that band has breathing room; column
  headers stay tight to the grids.
- Signed values keep their sign; comparisons use `|·|`.
- Chalk math for dib / `dib_diffs` extends to the **right** of `k` (still beside the counter).
- Stage scale: cells ~`CELL=1.2` with modest side/edge margins (fill the frame without crowding).
### Color language

- **Green / red** = cell **borders** only, on **input** and **`dib_diffs`** twins.
- Digits stay **white** (including `k`, amp, dib, chalk numerals).
- Center `amp|dib` cells stay neutral (no green/red strokes).
- Brief **gold** on `k` only during the increment+roll beat, then back to white.

---

## Act 0 — Arrive as a line

1. Empty stage; short title: “Linear Probing Amplitude Pooling”.
2. The 12 integers appear **left → right** in flat order (even spacing, no visual grouping), light dealt feel:

   `-1, 2, -8, 9, 6, 5, 7, -6, 3, 4, -5, -4`

   This order is **column-major on the 4×3 work grid** (fill each probe column top→bottom, columns left→right). Same numbers as the pedagogical grid; display order only.

3. Hold a beat so the viewer can read the strip.
4. Strip sits in the **upper** band of the blackboard (leave room for the stage below).

---

## Act 1 — Fold + put every table in place

Goal: finish **staging** before any LPAP projection. After this act, the full machine is visible and idle at `k=0`.

1. Annotate: `N=12 values   C=4 buckets   K=3 iterations`
   (`K` = `k_max` from the code; capital so it does not clash with the rolling `k`).
2. Materialize the empty **work** grid (4 rows × 3 cols) on the **left** (below the strip).
3. Fold: fill **vertically by columns**, left → right (matches strip order). Continuous motion is fine; no per-column highlight required.

   Final work grid:

   |    | p0 | p1 | p2 |
   |----|----|----|----|
   | B0 | -1 |  6 |  3 |
   | B1 |  2 |  5 |  4 |
   | B2 | -8 |  7 | -5 |
   | B3 |  9 | -6 | -4 |

4. Materialize the center **4×2** (amplitudes | dibs), all zeros, row-aligned with the work grid. Winners later slide **right** into the amplitude column; dibs update in the adjacent cell.
5. Materialize **`dib_diffs`** on the **right**, same 4×3 geometry, all zeros.
6. Show **`k=0`**.
7. Short hold: the board is fully set; LPAP beats start next.

Suggested appear order: annotation → work grid → fold chips → center 4×2 (amp|dib) → right `dib_diffs` → `k`.

---

## Act 2 — Projection at `k=0` (setup fill)

At `k=0`, bucket row `i` reads its own work row (identity).

### 2a. Highlight winners

For each row, stroke-highlight the cell with largest `|·|` **and its twin cell in `dib_diffs`**
(green if `|cand| ≥ |amp|`, red if it loses). Digits stay white; center panel stays neutral.
At `k=0` amp is empty (`0`), so every row max is green. Cascade top→bottom.

| Bucket | Winner | Slot | dibs |
|--------|--------|------|------|
| B0 | `6` (p1) | `6` | `0` |
| B1 | `5` (p1) | `5` | `0` |
| B2 | `-8` (p0) | `-8` | `0` |
| B3 | `9` (p0) | `9` | `0` |

### 2b. Slide into amp + swap-back

1. Each highlighted winner **slides right** from `input` into that row’s **amp** cell (left column of the center `amp|dib` panel).
2. The `0` that was in amp **swap-backs** along the same path into the vacated `input` cell.
3. Dib column stays `0` (idle at `k=0`; chalk for dibs writes starts at Act 4).
4. `dib_diffs` unchanged (all zeros). Act 2 chalk skipped — all zeros would be noisy.

After `k=0`:

```text
work:              amp dib     dib_diffs:
 -1   0   3         6   0        0 0 0
  2   0   4         5   0        0 0 0
  0   7  -5        -8   0        0 0 0
  0  -6  -4         9   0        0 0 0
```

Narration: first pass fills amp from empty buckets; no interesting displacements yet.

---

## Act 3 — Roll `0 → 1`

1. **`k → 1` in parallel with the roll**: update the counter (brief gold accent) **while** `input` and `dib_diffs` twin-roll, so the eye ties the increment to the wrap. Then `k` returns to white.
2. **Roll** `input` and `dib_diffs` together along the **row** axis (same motion on both, even when `dib_diffs` is all zeros — teaches the twin-roll principle):
   - Shift row contents **down by one**.
   - The bottom row **wraps to the top**.
3. Center **amp|dib** stay fixed. After this wrap, the `input` row beside amp-row-0 holds former row-3 content (`B0←L3` at `k=1`).

   Mapping: `B0←L3`, `B1←L0`, `B2←L1`, `B3←L2`.

After roll (before Act 4 swap):

```text
work:              amp dib     dib_diffs:
  0  -6  -4         6   0        0 0 0
 -1   0   3         5   0        0 0 0
  2   0   4        -8   0        0 0 0
  0   7  -5         9   0        0 0 0
```

---

## Act 4 — Projection at `k=1` (single swap — serial hero)

Stroke-highlight each row’s max `|·|` in the input **and the twin cell in `dib_diffs`**;
digits stay white; center/`k` stay neutral. **Do not dim** the rest of the board.
**Green** borders = beats amp (enters the table); **red** = loses to amp (no swap).
Only green candidates actually swap.

| Bucket | Reads | Candidate | vs table | Result |
|--------|-------|-----------|----------|--------|
| B0 | L3 row | `-6` | `\|-6\| ≥ \|6\|` | **swap** |
| B1 | L0 | `3` | `\|3\| < \|5\|` | highlight only |
| B2 | L1 | `4` | `\|4\| < \|-8\|` | highlight only |
| B3 | L2 | `7` | `\|7\| < \|9\|` | highlight only |

### Swap (slow)

1. Highlight every row’s max-|·| and its `dib_diffs` twin: green/red borders only.
2. Drop red on losers (input + twin) as the swap runs; leave center white.
3. Amp swap: `-6` ↔ `6`. In parallel, **duplicate `k`’s digit into the dib slot**; the previous dib (`0`) drops beside the counter.
4. Chalk under the stage (pedagogical): minus, then `=`, then the result `1`
   **appears in place** right of `=` (`k=1−0=1`); that `1` flies into the twin
   `dib_diffs` cell; leftover `−0=` fades.
   (Only the **dib copy** peels from `k`’s digit — the subtraction result does not.)

After `k=1`:

```text
work:              amp dib     dib_diffs:
  0   6  -4        -6   1        0  1  0
 -1   0   3         5   0        0  0  0
  2   0   4        -8   0        0  0  0
  0   7  -5         9   0        0  0  0
```

Hero: **`6`**, parked in input with `dib_diffs = +1` (`k − old_dib`).

---

## Act 5 — Roll `1 → 2`

Same choreography as Act 3:

1. **`k → 2` in parallel with the roll** (brief gold, then white).
2. Twin row-wrap on **input + `dib_diffs`**; center fixed.
3. The parked **`6`** and its twin **`+1`** ride together.

Mapping: `B0←L2`, `B1←L3`, `B2←L0`, `B3←L1`.

After roll (setup for Act 6):

```text
work:              amp dib     dib_diffs:
  0   7  -5        -6   1        0  0  0
  0   6  -4         5   0        0  1  0
 -1   0   3        -8   0        0  0  0
  2   0   4         9   0        0  0  0
```

---

## Act 6 — Projection at `k=2` (fresh + re-entry)

### 6a. Select — done

Stroke-highlight row maxima on input + `dib_diffs` (digits white; center/`k` neutral).
Leave borders up briefly for review.

| Bucket | Candidate | vs amp | Twin | Border |
|--------|-----------|--------|------|--------|
| B0 | `7` (p1) | `\|7\| ≥ \|-6\|` | `0` | **green** (fresh) |
| B1 | `6` (p1) | `\|6\| ≥ \|5\|` | `+1` | **green** (re-entry) |
| B2 | `3` (p2) | `\|3\| < \|-8\|` | `0` | **red** |
| B3 | `4` (p2) | `\|4\| < \|9\|` | `0` | **red** |

### 6b. Fresh B0 — done

Clone Act 4 chalk on B0 only; drop red losers during the swap; leave B1 green for 6c.

1. Amp swap: `7` ↔ `-6`.
2. Duplicate `k`’s digit into dib (`2`); old dib `1` drops beside the counter.
3. Chalk `k=2−1=1` with result appearing in place right of `=` → that `1` into twin
   `dib_diffs` (dib copy still peels from `k`).
4. Settle B0 borders; B1 stays green.

After 6b:

```text
work:              amp dib     dib_diffs:
  0  -6  -5         7   2        0  1  0
  0   6  -4         5   0        0  1  0
 -1   0   3        -8   0        0  0  0
  2   0   4         9   0        0  0  0
```

### 6c. Re-entry B1 — done

Real LPAP on every swap (including re-entry): **read** the candidate’s twin into
`dib`, then **overwrite** that same twin cell with a parking ticket for the
outgoing bucket value (`dib_diffs ← k − old_dib` pedagogically).

For B1 at `k=2`: incoming `6` has twin `+1` → `dib = 2 − 1 = 1`; outgoing `5`
has `old_dib = 0` → twin becomes `+2` (not `0`).

#### 6c-i / start of 6c-ii. Amp swap ‖ old dib → chalk

Same parallel language as Acts 4 / 6b: `6` ↔ `5` **while** old dib `0` drops
beside `k=2` into its final chalk seat. Twin still `+1` on that cell (now holding
`5`); dib slot vacant.

#### 6c-ii. Park the outgoing `5` (write twin `+2`)

1. Build chalk: `k=2−0=2` (result appears in place right of `=`).
2. **Swap with the twin:** rightmost `2` ↔ parked `+1` (2 into `dib_diffs`; `+1`
   into the chalk subtrahend, replacing `0`).
3. **Do not** fade minus/equals — the result seat is empty because that `2` moved.

Chalk now reads `k=2−1=`.

#### 6c-iii. Commit incoming dib

1. New result `1` appears after the equals (`dib = 2 − 1`).
2. That result flies into **dib**.
3. Fade leftover chalk; settle B1 borders.

After Act 6 (`k=2`):

```text
work:              amp dib     dib_diffs:
  0  -6  -5         7   2        0  1  0
  0   5  -4         6   1        0  2  0
 -1   0   3        -8   0        0  0  0
  2   0   4         9   0        0  0  0
```

---

## Act 7 — Reveal output — done

After Act 6 the machine has finished the `K=3` iterations (`k=0,1,2`). No more
rolls. The center **amp | dib** 4×2 is the LPAP **output**; everything else was
working memory.

### Beats

1. **Fade away the workspace** (together):  
   - `input` grid (cells + values + tag)  
   - `dib_diffs` grid (cells + values + tag)  
   - rolling `k` counter  
   Leave the center 4×2 (`amp` / `dib` + values) and the top annotation
   (`N` / `C` / `K`) on screen.
2. **Gold flash on the center** — same accent as the `k` increment: pulse cell
   strokes, amp/dib tags, **and the amp/dib digits**, then settle back to neutral
   (digits white again).
3. Hold on the isolated output table.

---

## Motion / pedagogy rules

1. One idea per beat until Act 6; Act 6 stages fresh then re-entry serially.
2. Work grid and `dib_diffs` are twins (roll + cell writes).
3. Selection moves **horizontally** (row → center bucket); swap-backs reverse on the same path.
4. On swap: chalk `dib_diffs = k − old_dib` into the twin (may displace a prior twin
   value). On re-entry, that recovered twin feeds chalk for `dib = k − twin`, then
   a new result flies into dib — do not clear the twin to `0`.
5. Keep `k` readable; brief gold only on increment+roll, then white. Act 7 reuses
   that gold once on the center output.
6. Prefer row-wise attention before committing a winner.
7. Act 1 must place **all** chrome (work, buckets, dibs, `dib_diffs`, `k`) before Act 2.
8. Green/red = side-panel **borders** only; digits always white.
9. Act 7 teaches “center = output” by subtraction (fade workspace) + gold flash —
   not by renaming panels mid-film.

---

## Timing sketch (rough)

| Act | Content | ~seconds |
|-----|---------|----------|
| 0 | Line appear | 4–6 |
| 1 | Fold + place all tables + `k` + `K=3` | 8–12 |
| 2 | `k=0` fills | 8–12 |
| 3 | Roll to 1 | 3–4 |
| 4 | Single swap + `dib_diffs=+1` chalk | 8–12 |
| 5 | Roll to 2 (`+1` rides with `6`) | 3–4 |
| 6 | Fresh + re-entry | 12–16 |
| 7 | Fade workspace; gold-flash output | 4–6 |
| | **Total** | **~55–70** |

---

## Open choices (when coding Act 7 / Act 1 chrome)

- Annotation layout if `N=… C=… K=…` feels wide (one line vs wrap).
- Fade: simultaneous vs input → `dib_diffs` → `k`.
- Gold flash: strokes only vs strokes + tag pulse; one pulse vs two.
- Whether title stays visible through Act 7.
- Voiceover vs captions only.
