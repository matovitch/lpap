"""LPAP operator Manim scenes (built act by act against the storyboard)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from manim import (
    DOWN,
    GOLD,
    GREEN,
    GREY_A,
    GREY_B,
    LEFT,
    ORIGIN,
    RED,
    RIGHT,
    UP,
    WHITE,
    AnimationGroup,
    Create,
    FadeIn,
    FadeOut,
    LaggedStart,
    ReplacementTransform,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

# Pedagogical grid: row = bucket, col = probe (p0..p2 left→right).
_ROWS: tuple[tuple[int, ...], ...] = (
    (-1, 6, 3),
    (2, 5, 4),
    (-8, 7, -5),
    (9, -6, -4),
)
C = len(_ROWS)
P = len(_ROWS[0])
N = C * P
K_MAX = 3  # on-screen capital K; matches code k_max
# Strip / fold order: column-major (each probe column top→bottom, L→R).
FLAT_VALUES: tuple[int, ...] = tuple(
    _ROWS[row][col] for col in range(P) for row in range(C)
)
# (row, col) of max-|·| winners at k=0 (identity read).
K0_WINNERS: tuple[tuple[int, int], ...] = (
    (0, 1),  # 6
    (1, 1),  # 5
    (2, 0),  # -8
    (3, 0),  # 9
)
# Single serial swap at k=1 after first roll.
# Pedagogical dibs_diff write is k - old_dib (= +1 here); not the real-code sign.
K1_SWAP: tuple[int, int] = (0, 1)  # -6 displaces 6; dibs_diff ← +1
# Fresh swap at k=2 (after Act 5 roll): 7 displaces -6; chalk 2−1 → twin +1.
K2_FRESH: tuple[int, int] = (0, 1)
# Re-entry at k=2: 6 displaces 5; consume twin +1 → dib = 2−1 = 1.
K2_REENTRY: tuple[int, int] = (1, 1)
CELL = 1.2
CHIP_FONT = 42
STACK_FONT = 38
LABEL_FONT = 28
TITLE_FONT = 40
ANNOTATION_FONT = 28
K_FONT = 38
HIGHLIGHT = GOLD
WINNER = GREEN
LOSER = RED

# Blackboard: strip high; stage below as work | amp+dib | dib_diffs.
# Stage low in the frame (tight bottom margin) so k / Act 0 strip breathe
# between N/C/K and the column headers (headers stay tight to the grids).
STAGE_Y = 1.1 * DOWN
GAP = 0.28
WORK_WIDTH = P * CELL
CENTER_WIDTH = 2 * CELL
DIFF_WIDTH = P * CELL
STAGE_WIDTH = WORK_WIDTH + CENTER_WIDTH + DIFF_WIDTH + 2 * GAP
WORK_CENTER = ORIGIN + STAGE_Y + (STAGE_WIDTH / 2 - WORK_WIDTH / 2) * LEFT
CENTER_CENTER = WORK_CENTER + ((WORK_WIDTH + CENTER_WIDTH) / 2 + GAP) * RIGHT
DIFF_CENTER = CENTER_CENTER + ((CENTER_WIDTH + DIFF_WIDTH) / 2 + GAP) * RIGHT
ANNOTATION_BUFF = 0.16


def _value_label(value: int, *, font_size: int = CHIP_FONT) -> Text:
    return Text(f"{value:2d}", font_size=font_size)


def _flat_chips(*, jitter: bool = True, center=ORIGIN) -> VGroup:
    chips = VGroup(*[_value_label(value) for value in FLAT_VALUES])
    chips.arrange(RIGHT, buff=0.22)
    chips.move_to(center)
    if jitter:
        rng = random.Random(0)
        for chip in chips:
            chip.shift(0.04 * rng.uniform(-1.0, 1.0) * UP)
    return chips


def _cell_point(row: int, col: int, *, cols: int, center):
    x = (col - (cols - 1) / 2) * CELL
    y = ((C - 1) / 2 - row) * CELL
    return center + x * RIGHT + y * UP


def _make_cell(*, color=GREY_B) -> RoundedRectangle:
    return RoundedRectangle(
        width=CELL * 0.90,
        height=CELL * 0.90,
        corner_radius=0.07,
        color=color,
        stroke_width=2,
    )


def _grid_cells(*, rows: int, cols: int, center, color=GREY_B) -> VGroup:
    cells = VGroup()
    for row in range(rows):
        for col in range(cols):
            cell = _make_cell(color=color)
            cell.move_to(_cell_point(row, col, cols=cols, center=center))
            cells.add(cell)
    return cells


def _column_header(text: str, *, font_size: int = LABEL_FONT, color=GREY_B) -> Text:
    """Header with uniform metrics via invisible H…y letter struts (plain Text)."""
    mob = Text(f"H{text}y", font_size=font_size, color=color)
    mob[0].set_opacity(0.0)
    mob[-1].set_opacity(0.0)
    return mob


def _headers_top_y() -> float:
    """Top of column-header band (matches Act 1 header placement)."""
    top_row_center_y = WORK_CENTER[1] + ((C - 1) / 2) * CELL
    cell_top = top_row_center_y + 0.45 * CELL
    header_baseline_y = cell_top + 0.12
    # H…y strut for LABEL_FONT sits roughly this tall above the baseline.
    return header_baseline_y + 0.40


def _annotation_bottom_y(title: Text) -> float:
    """Bottom of the N/C/K line if placed under title (without adding it)."""
    probe = Text(
        f"N={N} values   C={C} buckets   K={K_MAX} iterations",
        font_size=ANNOTATION_FONT,
    )
    probe.next_to(title, DOWN, buff=ANNOTATION_BUFF)
    return float(probe.get_bottom()[1])


def _k_band_center_y(title: Text) -> float:
    """Vertical center for k and the Act 0 strip (between annotation and headers)."""
    return 0.5 * (_annotation_bottom_y(title) + _headers_top_y())


def _place_column_header(header: Text, *, target_x: float, baseline_y: float) -> None:
    """Center visible glyphs on target_x; align H-bottom to a shared baseline_y."""
    visible = VGroup(*header[1:-1])
    header.shift(
        [
            target_x - visible.get_center()[0],
            baseline_y - header[0].get_bottom()[1],
            0.0,
        ]
    )


def _place_k_counter(label: Text, *, band_y: float) -> None:
    """Park k centered horizontally on the shared strip/k band."""
    stage_x = 0.5 * (WORK_CENTER[0] + DIFF_CENTER[0])
    label.move_to([stage_x, band_y, 0.0])


def _work_cell(work_cells: VGroup, row: int, col: int) -> RoundedRectangle:
    return work_cells[row * P + col]


@dataclass
class StageState:
    chips: VGroup
    work_cells: VGroup
    center_cells: VGroup
    diff_cells: VGroup
    amp_labels: VGroup
    dib_labels: VGroup
    diff_labels: VGroup
    input_tag: Text
    amp_tag: Text
    dib_tag: Text
    diff_tag: Text
    k_label: Text
    # Occupants after layout / swaps (row-major logical grids).
    work_values: list[list[Text]]
    amp_values: list[Text]
    dib_values: list[Text]
    diff_values: list[list[Text]]


def play_act0_arrive(scene: Scene, *, hold: float = 1.6) -> tuple[Text, VGroup]:
    title = Text("Linear Probing Amplitude Pooling", font_size=TITLE_FONT)
    title.to_edge(UP, buff=0.18)
    # Same vertical band as k later (between N/C/K and table headers).
    strip_center = ORIGIN + _k_band_center_y(title) * UP
    chips = _flat_chips(jitter=True, center=strip_center)

    scene.play(Write(title), run_time=1.0)
    scene.wait(0.2)
    scene.play(
        LaggedStart(
            *[FadeIn(chip, shift=0.2 * DOWN) for chip in chips],
            lag_ratio=0.16,
        ),
        run_time=2.6,
    )
    scene.wait(hold)
    return title, chips


def play_act1_stage(scene: Scene, *, hold_end: float = 1.0) -> StageState:
    """Fold into 4×3 input and place amp|dib, dibs_diff, and k=0."""
    title, chips = play_act0_arrive(scene, hold=0.45)

    annotation = Text(
        f"N={N} values   C={C} buckets   K={K_MAX} iterations",
        font_size=ANNOTATION_FONT,
    )
    annotation.next_to(title, DOWN, buff=ANNOTATION_BUFF)

    work_cells = _grid_cells(rows=C, cols=P, center=WORK_CENTER)
    center_cells = _grid_cells(rows=C, cols=2, center=CENTER_CENTER, color=GREY_A)
    diff_cells = _grid_cells(rows=C, cols=P, center=DIFF_CENTER)

    amp_labels = VGroup()
    dib_labels = VGroup()
    for row in range(C):
        amp = _value_label(0, font_size=STACK_FONT)
        dib = _value_label(0, font_size=STACK_FONT)
        amp.move_to(center_cells[row * 2].get_center())
        dib.move_to(center_cells[row * 2 + 1].get_center())
        amp_labels.add(amp)
        dib_labels.add(dib)

    diff_labels = VGroup()
    for cell in diff_cells:
        label = _value_label(0, font_size=STACK_FONT)
        label.move_to(cell.get_center())
        diff_labels.add(label)

    header_baseline_y = (
        max(
            work_cells.get_top()[1],
            center_cells.get_top()[1],
            diff_cells.get_top()[1],
        )
        + 0.12
    )
    input_tag = _column_header("input")
    amp_tag = _column_header("amp")
    dib_tag = _column_header("dib")
    diff_tag = _column_header("dib_diffs")
    _place_column_header(input_tag, target_x=WORK_CENTER[0], baseline_y=header_baseline_y)
    _place_column_header(
        amp_tag,
        target_x=center_cells[0].get_center()[0],
        baseline_y=header_baseline_y,
    )
    _place_column_header(
        dib_tag,
        target_x=center_cells[1].get_center()[0],
        baseline_y=header_baseline_y,
    )
    _place_column_header(diff_tag, target_x=DIFF_CENTER[0], baseline_y=header_baseline_y)

    k_label = Text("k=0", font_size=K_FONT)
    _place_k_counter(k_label, band_y=_k_band_center_y(title))

    scene.play(FadeIn(annotation), run_time=0.55)
    scene.wait(0.1)

    scene.play(
        LaggedStart(*[Create(cell) for cell in work_cells], lag_ratio=0.03),
        FadeIn(input_tag),
        run_time=0.9,
    )
    scene.wait(0.1)

    fold_moves = [
        chips[col * C + row].animate.move_to(work_cells[row * P + col].get_center())
        for col in range(P)
        for row in range(C)
    ]
    scene.play(LaggedStart(*fold_moves, lag_ratio=0.10), run_time=1.6)
    scene.wait(0.15)

    scene.play(
        LaggedStart(*[Create(cell) for cell in center_cells], lag_ratio=0.04),
        FadeIn(amp_tag),
        FadeIn(dib_tag),
        run_time=0.8,
    )
    scene.play(
        LaggedStart(
            *[FadeIn(label) for label in (*amp_labels, *dib_labels)],
            lag_ratio=0.05,
        ),
        run_time=0.7,
    )
    scene.wait(0.1)

    scene.play(
        LaggedStart(*[Create(cell) for cell in diff_cells], lag_ratio=0.03),
        FadeIn(diff_tag),
        run_time=0.8,
    )
    scene.play(
        LaggedStart(*[FadeIn(label) for label in diff_labels], lag_ratio=0.03),
        run_time=0.6,
    )
    scene.wait(0.1)

    scene.play(FadeIn(k_label, shift=0.15 * UP), run_time=0.45)
    scene.wait(hold_end)

    work_values = [
        [chips[col * C + row] for col in range(P)] for row in range(C)
    ]
    amp_values = [amp_labels[row] for row in range(C)]
    dib_values = [dib_labels[row] for row in range(C)]
    diff_values = [
        [diff_labels[row * P + col] for col in range(P)] for row in range(C)
    ]

    return StageState(
        chips=chips,
        work_cells=work_cells,
        center_cells=center_cells,
        diff_cells=diff_cells,
        amp_labels=amp_labels,
        dib_labels=dib_labels,
        diff_labels=diff_labels,
        input_tag=input_tag,
        amp_tag=amp_tag,
        dib_tag=dib_tag,
        diff_tag=diff_tag,
        k_label=k_label,
        work_values=work_values,
        amp_values=amp_values,
        dib_values=dib_values,
        diff_values=diff_values,
    )


def play_act2_project_k0(scene: Scene, stage: StageState, *, hold_end: float = 1.4) -> None:
    """Highlight max-|·| per row (green if beats amp, else red); swap winners in."""
    # Same compare rule as later k: green enters amp, red stays put.
    row_max_cols = []
    beats_amp = []
    for row in range(C):
        values = stage.work_values[row]
        best_col = max(range(P), key=lambda c: abs(int(values[c].text)))
        row_max_cols.append(best_col)
        cand_abs = abs(int(values[best_col].text))
        amp_abs = abs(int(stage.amp_values[row].text))
        beats_amp.append(cand_abs >= amp_abs)

    # At k=0 amp is empty (0), so every row max should win.
    assert all(beats_amp)
    assert all(
        (row, row_max_cols[row]) == expected
        for row, expected in enumerate(K0_WINNERS)
    )

    candidates = [stage.work_values[row][row_max_cols[row]] for row in range(C)]
    candidate_cells = [
        _work_cell(stage.work_cells, row, row_max_cols[row]) for row in range(C)
    ]
    twin_cells = [
        stage.diff_cells[row * P + row_max_cols[row]] for row in range(C)
    ]
    amp_zeros = list(stage.amp_values)
    work_targets = [cell.get_center() for cell in candidate_cells]
    amp_targets = [stage.center_cells[row * 2].get_center() for row in range(C)]

    # Green/red on input + dibs_diff borders only; digits stay white.
    highlight = []
    for cell, twin, win in zip(candidate_cells, twin_cells, beats_amp, strict=True):
        color = WINNER if win else LOSER
        highlight.append(
            AnimationGroup(
                cell.animate.set_stroke(color, width=4),
                twin.animate.set_stroke(color, width=4),
            )
        )
    scene.play(LaggedStart(*highlight, lag_ratio=0.22), run_time=1.4)
    scene.wait(0.35)

    # Slide right into amp; zeros swap-back into input along the same paths.
    winners = [c for c, win in zip(candidates, beats_amp, strict=True) if win]
    winner_cells = [c for c, win in zip(candidate_cells, beats_amp, strict=True) if win]
    winner_twins = [c for c, win in zip(twin_cells, beats_amp, strict=True) if win]
    winner_zeros = [z for z, win in zip(amp_zeros, beats_amp, strict=True) if win]
    winner_work_pos = [p for p, win in zip(work_targets, beats_amp, strict=True) if win]
    winner_amp_pos = [p for p, win in zip(amp_targets, beats_amp, strict=True) if win]

    scene.play(
        LaggedStart(
            *[
                AnimationGroup(
                    chip.animate.move_to(amp_pos),
                    zero.animate.move_to(work_pos),
                )
                for chip, zero, amp_pos, work_pos in zip(
                    winners,
                    winner_zeros,
                    winner_amp_pos,
                    winner_work_pos,
                    strict=True,
                )
            ],
            lag_ratio=0.18,
        ),
        run_time=2.0,
    )
    scene.wait(0.25)

    scene.play(
        *[cell.animate.set_stroke(GREY_B, width=2) for cell in winner_cells],
        *[twin.animate.set_stroke(GREY_B, width=2) for twin in winner_twins],
        run_time=0.6,
    )

    for row, col, chip, zero, win in zip(
        range(C), row_max_cols, candidates, amp_zeros, beats_amp, strict=True
    ):
        if not win:
            continue
        stage.work_values[row][col] = zero
        stage.amp_values[row] = chip

    scene.wait(hold_end)


def play_twin_roll(
    scene: Scene,
    stage: StageState,
    *,
    k_value: int,
    hold_end: float = 1.2,
) -> None:
    """Twin row-roll on input & dibs_diff, with k→k_value in the same beat."""
    new_k = Text(f"k={k_value}", font_size=K_FONT, color=HIGHLIGHT)
    new_k.move_to(stage.k_label.get_center())

    moves = []
    for row in range(C):
        for col in range(P):
            new_row = (row + 1) % C
            work_mob = stage.work_values[row][col]
            diff_mob = stage.diff_values[row][col]
            moves.append(
                work_mob.animate.move_to(
                    _work_cell(stage.work_cells, new_row, col).get_center()
                )
            )
            moves.append(
                diff_mob.animate.move_to(
                    stage.diff_cells[new_row * P + col].get_center()
                )
            )

    # One beat: k increments while both matrices wrap (ties counter to roll).
    scene.play(
        ReplacementTransform(stage.k_label, new_k),
        *moves,
        run_time=1.9,
    )
    stage.k_label = new_k
    stage.work_values = [
        [stage.work_values[(row - 1) % C][col] for col in range(P)]
        for row in range(C)
    ]
    stage.diff_values = [
        [stage.diff_values[(row - 1) % C][col] for col in range(P)]
        for row in range(C)
    ]
    scene.play(stage.k_label.animate.set_color(WHITE), run_time=0.4)
    scene.wait(hold_end)


def play_act3_roll_k1(scene: Scene, stage: StageState, *, hold_end: float = 1.2) -> None:
    """Act 3: roll 0→1."""
    play_twin_roll(scene, stage, k_value=1, hold_end=hold_end)


def play_act5_roll_k2(scene: Scene, stage: StageState, *, hold_end: float = 1.2) -> None:
    """Act 5: roll 1→2 (+1 rides with parked 6)."""
    play_twin_roll(scene, stage, k_value=2, hold_end=hold_end)


@dataclass
class SelectState:
    """Per-row max-|·| selection vs current amp (borders only)."""

    row_max_cols: list[int]
    beats_amp: list[bool]
    candidate_cells: list[RoundedRectangle]
    twin_cells: list[RoundedRectangle]


def play_select_borders(
    scene: Scene,
    stage: StageState,
    *,
    lag_ratio: float = 0.18,
    run_time: float = 1.2,
    hold: float = 0.35,
) -> SelectState:
    """Stroke-highlight each row max on input + dibs_diff (green win / red lose)."""
    row_max_cols: list[int] = []
    beats_amp: list[bool] = []
    for row in range(C):
        values = stage.work_values[row]
        best_col = max(range(P), key=lambda c: abs(int(values[c].text)))
        row_max_cols.append(best_col)
        cand_abs = abs(int(values[best_col].text))
        amp_abs = abs(int(stage.amp_values[row].text))
        beats_amp.append(cand_abs >= amp_abs)

    candidate_cells = [
        _work_cell(stage.work_cells, row, row_max_cols[row]) for row in range(C)
    ]
    twin_cells = [
        stage.diff_cells[row * P + row_max_cols[row]] for row in range(C)
    ]

    highlight = []
    for cell, twin, win in zip(candidate_cells, twin_cells, beats_amp, strict=True):
        color = WINNER if win else LOSER
        highlight.append(
            AnimationGroup(
                cell.animate.set_stroke(color, width=4),
                twin.animate.set_stroke(color, width=4),
            )
        )
    scene.play(LaggedStart(*highlight, lag_ratio=lag_ratio), run_time=run_time)
    scene.wait(hold)
    return SelectState(
        row_max_cols=row_max_cols,
        beats_amp=beats_amp,
        candidate_cells=candidate_cells,
        twin_cells=twin_cells,
    )


# Expected Act 6 select after the k=2 roll: fresh 7 + re-entry 6 win; 3 and 4 lose.
K2_SELECT: tuple[tuple[int, int, bool], ...] = (
    (0, 1, True),   # 7
    (1, 1, True),   # 6 with twin +1
    (2, 2, False),  # 3
    (3, 2, False),  # 4
)


def play_act6a_select_k2(
    scene: Scene, stage: StageState, *, hold_end: float = 1.6
) -> SelectState:
    """Act 6a: border-select at k=2; leave greens/reds up for review."""
    selected = play_select_borders(scene, stage, lag_ratio=0.18, run_time=1.3, hold=0.4)
    for row, (exp_row, exp_col, exp_win) in enumerate(K2_SELECT):
        assert row == exp_row
        assert selected.row_max_cols[row] == exp_col
        assert selected.beats_amp[row] is exp_win
    scene.wait(hold_end)
    return selected


def _k_value(stage: StageState) -> int:
    return int(stage.k_label.text.split("=")[-1].strip())


def play_fresh_swap_chalk(
    scene: Scene,
    stage: StageState,
    *,
    row: int,
    col: int,
    selected: SelectState,
    settle_winner_borders: bool = True,
    keep_green_rows: frozenset[int] = frozenset(),
    hold_end: float = 1.0,
) -> None:
    """Amp↔candidate swap; k→dib; chalk dibs_diff = k − old_dib into the twin."""
    assert selected.row_max_cols[row] == col and selected.beats_amp[row]
    k_value = _k_value(stage)
    old_dib = stage.dib_values[row]
    old_dib_value = int(old_dib.text)
    diff_value = k_value - old_dib_value

    candidate = stage.work_values[row][col]
    amp_chip = stage.amp_values[row]
    old_diff = stage.diff_values[row][col]
    work_cell = selected.candidate_cells[row]
    amp_cell = stage.center_cells[row * 2]
    dib_cell = stage.center_cells[row * 2 + 1]
    diff_cell = selected.twin_cells[row]
    k_digit = stage.k_label[-1]

    k_to_dib = _value_label(k_value, font_size=STACK_FONT)
    k_to_dib.move_to(k_digit.get_center())
    scene.add(k_to_dib)

    minus = Text("−", font_size=K_FONT)
    equals = Text("=", font_size=K_FONT)
    result = _value_label(diff_value, font_size=STACK_FONT)
    minus.next_to(stage.k_label, RIGHT, buff=0.14)
    chalk_probe = old_dib.copy()
    chalk_probe.next_to(minus, RIGHT, buff=0.14)
    chalk_old_pos = chalk_probe.get_center()

    lose_settle = []
    for other in range(C):
        if selected.beats_amp[other]:
            continue
        lose_settle.append(
            selected.candidate_cells[other].animate.set_stroke(GREY_B, width=2)
        )
        lose_settle.append(
            selected.twin_cells[other].animate.set_stroke(GREY_B, width=2)
        )

    scene.play(
        candidate.animate.move_to(amp_cell.get_center()),
        amp_chip.animate.move_to(work_cell.get_center()),
        k_to_dib.animate.move_to(dib_cell.get_center()),
        old_dib.animate.move_to(chalk_old_pos),
        *lose_settle,
        run_time=1.9,
    )
    stage.work_values[row][col] = amp_chip
    stage.amp_values[row] = candidate
    stage.dib_values[row] = k_to_dib
    scene.wait(0.25)

    scene.play(FadeIn(minus, scale=0.8), run_time=0.55)
    equals.next_to(old_dib, RIGHT, buff=0.14)
    scene.play(FadeIn(equals), run_time=0.45)
    # Subtraction result appears in place (right of =) — not peeled from k's digit.
    result.next_to(equals, RIGHT, buff=0.14)
    scene.play(FadeIn(result, scale=0.85), run_time=0.55)
    scene.wait(0.35)

    scene.play(
        result.animate.move_to(diff_cell.get_center()),
        FadeOut(old_diff, scale=0.6),
        run_time=1.3,
    )
    scene.remove(old_diff)
    stage.diff_values[row][col] = result
    scene.wait(0.35)

    settle = [FadeOut(minus), FadeOut(old_dib), FadeOut(equals)]
    if settle_winner_borders and row not in keep_green_rows:
        settle.append(work_cell.animate.set_stroke(GREY_B, width=2))
        settle.append(diff_cell.animate.set_stroke(GREY_B, width=2))
    scene.play(*settle, run_time=0.7)
    scene.wait(hold_end)


def play_act4_project_k1(scene: Scene, stage: StageState, *, hold_end: float = 1.6) -> None:
    """At k=1: green/red select; swap winner; chalk k−old_dib into dibs_diff."""
    selected = play_select_borders(scene, stage, lag_ratio=0.18, run_time=1.2, hold=0.35)
    row, col = K1_SWAP
    assert selected.row_max_cols[row] == col and selected.beats_amp[row]
    assert sum(selected.beats_amp) == 1
    play_fresh_swap_chalk(
        scene,
        stage,
        row=row,
        col=col,
        selected=selected,
        settle_winner_borders=True,
        hold_end=hold_end,
    )


def play_act6b_fresh_b0(
    scene: Scene,
    stage: StageState,
    selected: SelectState,
    *,
    hold_end: float = 1.4,
) -> None:
    """Act 6b: fresh B0 — 7 displaces -6; chalk 2−1 into twin; leave B1 green."""
    row, col = K2_FRESH
    play_fresh_swap_chalk(
        scene,
        stage,
        row=row,
        col=col,
        selected=selected,
        settle_winner_borders=True,
        keep_green_rows=frozenset({1}),  # re-entry still pending
        hold_end=hold_end,
    )


def play_reentry_swap_chalk(
    scene: Scene,
    stage: StageState,
    *,
    row: int,
    col: int,
    selected: SelectState,
    settle_winner_borders: bool = True,
    hold_end: float = 1.0,
) -> None:
    """Re-entry: amp swap; park outgoing (twin←k−old_dib); dib←k−recovered twin."""
    assert selected.row_max_cols[row] == col and selected.beats_amp[row]
    k_value = _k_value(stage)
    twin_chip = stage.diff_values[row][col]  # incoming's parked +1
    twin_value = int(twin_chip.text)
    old_dib = stage.dib_values[row]
    old_dib_value = int(old_dib.text)
    park_value = k_value - old_dib_value  # pedagogical ticket for outgoing
    dib_value = k_value - twin_value  # dib from recovered twin
    assert park_value == 2 and twin_value == 1 and dib_value == 1

    candidate = stage.work_values[row][col]
    amp_chip = stage.amp_values[row]
    work_cell = selected.candidate_cells[row]
    amp_cell = stage.center_cells[row * 2]
    dib_cell = stage.center_cells[row * 2 + 1]
    diff_cell = selected.twin_cells[row]

    # Pre-layout chalk so old dib can fly in parallel with the amp↔input swap
    # (same timing language as Acts 4 / 6b).
    minus = Text("−", font_size=K_FONT)
    equals = Text("=", font_size=K_FONT)
    park_result = _value_label(park_value, font_size=STACK_FONT)
    minus.next_to(stage.k_label, RIGHT, buff=0.14)
    chalk_probe = old_dib.copy()
    chalk_probe.next_to(minus, RIGHT, buff=0.14)
    chalk_old_pos = chalk_probe.get_center()

    # --- 6c-i + start of 6c-ii: amp swap ‖ old dib → chalk ---
    scene.play(
        candidate.animate.move_to(amp_cell.get_center()),
        amp_chip.animate.move_to(work_cell.get_center()),
        old_dib.animate.move_to(chalk_old_pos),
        run_time=1.9,
    )
    stage.work_values[row][col] = amp_chip
    stage.amp_values[row] = candidate
    scene.wait(0.25)

    # --- 6c-ii continued: chalk k−old_dib=park; swap into twin ---
    scene.play(FadeIn(minus, scale=0.8), run_time=0.55)
    equals.next_to(old_dib, RIGHT, buff=0.14)
    scene.play(FadeIn(equals), run_time=0.45)
    # Park result appears in place (right of =) — not peeled from k's digit.
    park_result.next_to(equals, RIGHT, buff=0.14)
    scene.play(FadeIn(park_result, scale=0.85), run_time=0.55)
    scene.wait(0.25)

    # Twin +1 ↔ chalk park 2: 2 into dib_diffs, +1 into subtrahend seat (replaces 0).
    scene.play(
        park_result.animate.move_to(diff_cell.get_center()),
        twin_chip.animate.move_to(chalk_old_pos),
        FadeOut(old_dib, scale=0.5),
        run_time=1.5,
    )
    scene.remove(old_dib)
    stage.diff_values[row][col] = park_result
    # dib cell vacant until 6c-iii; recovered twin sits in chalk as subtrahend.
    scene.wait(0.3)

    # --- 6c-iii: new dib result appears after =, flies into dib ---
    dib_result = _value_label(dib_value, font_size=STACK_FONT)
    dib_probe = dib_result.copy()
    dib_probe.next_to(equals, RIGHT, buff=0.14)
    dib_chalk_target = dib_probe.get_center()
    dib_result.move_to(dib_chalk_target)
    dib_result.set_opacity(0.0)
    scene.add(dib_result)
    scene.play(dib_result.animate.set_opacity(1.0), run_time=0.45)
    scene.wait(0.25)

    scene.play(dib_result.animate.move_to(dib_cell.get_center()), run_time=1.2)
    stage.dib_values[row] = dib_result
    scene.wait(0.3)

    settle = [
        FadeOut(minus),
        FadeOut(twin_chip),
        FadeOut(equals),
    ]
    if settle_winner_borders:
        settle.append(work_cell.animate.set_stroke(GREY_B, width=2))
        settle.append(diff_cell.animate.set_stroke(GREY_B, width=2))
    scene.play(*settle, run_time=0.7)
    scene.wait(hold_end)


def play_act6c_reentry_b1(
    scene: Scene,
    stage: StageState,
    selected: SelectState,
    *,
    hold_end: float = 1.4,
) -> None:
    """Act 6c: re-entry B1 — park outgoing twin +2; dib ← k − recovered +1."""
    row, col = K2_REENTRY
    assert int(stage.diff_values[row][col].text) == 1
    assert int(stage.dib_values[row].text) == 0
    play_reentry_swap_chalk(
        scene,
        stage,
        row=row,
        col=col,
        selected=selected,
        settle_winner_borders=True,
        hold_end=hold_end,
    )


class Act0Arrive(Scene):
    """Act 0: flat integers appear left → right with a light dealt feel."""

    def construct(self) -> None:
        play_act0_arrive(self, hold=1.6)


class Act1Fold(Scene):
    """Act 1: fold into 4×3 work grid and place amp|dib + dibs_diff + k."""

    def construct(self) -> None:
        play_act1_stage(self, hold_end=1.6)


class Act2ProjectK0(Scene):
    """Act 2: k=0 max-|·| highlight, slide into amp, swap-back."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.5)
        play_act2_project_k0(self, stage, hold_end=1.4)


class Act3RollK1(Scene):
    """Act 3: k→1 flash, then twin row-roll on input and dibs_diff."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.35)
        play_act2_project_k0(self, stage, hold_end=0.45)
        play_act3_roll_k1(self, stage, hold_end=1.2)


class Act4ProjectK1(Scene):
    """Act 4: serial hero swap at k=1 + first nonzero dibs_diff."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.3)
        play_act2_project_k0(self, stage, hold_end=0.35)
        play_act3_roll_k1(self, stage, hold_end=0.4)
        play_act4_project_k1(self, stage, hold_end=1.5)


class Act5RollK2(Scene):
    """Act 5: k→2 + twin row-roll; +1 rides with parked 6."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.25)
        play_act2_project_k0(self, stage, hold_end=0.3)
        play_act3_roll_k1(self, stage, hold_end=0.35)
        play_act4_project_k1(self, stage, hold_end=0.4)
        play_act5_roll_k2(self, stage, hold_end=1.4)


class Act6aSelectK2(Scene):
    """Act 6a: at k=2, green/red border select (fresh 7 + re-entry 6)."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.2)
        play_act2_project_k0(self, stage, hold_end=0.25)
        play_act3_roll_k1(self, stage, hold_end=0.3)
        play_act4_project_k1(self, stage, hold_end=0.3)
        play_act5_roll_k2(self, stage, hold_end=0.35)
        play_act6a_select_k2(self, stage, hold_end=1.8)


class Act6bFreshB0(Scene):
    """Act 6b: fresh B0 chalk swap (7 in, dib←2, twin←1); B1 stays green."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.15)
        play_act2_project_k0(self, stage, hold_end=0.2)
        play_act3_roll_k1(self, stage, hold_end=0.25)
        play_act4_project_k1(self, stage, hold_end=0.25)
        play_act5_roll_k2(self, stage, hold_end=0.3)
        selected = play_act6a_select_k2(self, stage, hold_end=0.35)
        play_act6b_fresh_b0(self, stage, selected, hold_end=1.6)


class Act6cReentryB1(Scene):
    """Act 6c: re-entry B1 — consume twin +1 into dib via k−diff chalk."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.12)
        play_act2_project_k0(self, stage, hold_end=0.18)
        play_act3_roll_k1(self, stage, hold_end=0.2)
        play_act4_project_k1(self, stage, hold_end=0.2)
        play_act5_roll_k2(self, stage, hold_end=0.25)
        selected = play_act6a_select_k2(self, stage, hold_end=0.25)
        play_act6b_fresh_b0(self, stage, selected, hold_end=0.3)
        play_act6c_reentry_b1(self, stage, selected, hold_end=1.6)


class Act6ProjectK2(Scene):
    """Full Act 6: select, fresh B0, re-entry B1."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.12)
        play_act2_project_k0(self, stage, hold_end=0.18)
        play_act3_roll_k1(self, stage, hold_end=0.2)
        play_act4_project_k1(self, stage, hold_end=0.2)
        play_act5_roll_k2(self, stage, hold_end=0.25)
        selected = play_act6a_select_k2(self, stage, hold_end=0.25)
        play_act6b_fresh_b0(self, stage, selected, hold_end=0.25)
        play_act6c_reentry_b1(self, stage, selected, hold_end=1.4)


def play_act7_reveal_output(
    scene: Scene, stage: StageState, *, hold_end: float = 1.8
) -> None:
    """Fade workspace; gold-flash center amp|dib as the LPAP output."""
    workspace = VGroup(
        stage.work_cells,
        stage.input_tag,
        *[chip for row in stage.work_values for chip in row],
        stage.diff_cells,
        stage.diff_tag,
        *[chip for row in stage.diff_values for chip in row],
        stage.k_label,
    )
    scene.play(FadeOut(workspace), run_time=1.5)
    scene.wait(0.4)

    # Same gold as k-increment; flash cells, tags, and amp/dib digits.
    scene.play(
        *[cell.animate.set_stroke(HIGHLIGHT, width=4) for cell in stage.center_cells],
        stage.amp_tag.animate.set_color(HIGHLIGHT),
        stage.dib_tag.animate.set_color(HIGHLIGHT),
        *[chip.animate.set_color(HIGHLIGHT) for chip in stage.amp_values],
        *[chip.animate.set_color(HIGHLIGHT) for chip in stage.dib_values],
        run_time=0.75,
    )
    scene.wait(0.55)
    scene.play(
        *[cell.animate.set_stroke(GREY_A, width=2) for cell in stage.center_cells],
        stage.amp_tag.animate.set_color(GREY_B),
        stage.dib_tag.animate.set_color(GREY_B),
        *[chip.animate.set_color(WHITE) for chip in stage.amp_values],
        *[chip.animate.set_color(WHITE) for chip in stage.dib_values],
        run_time=0.65,
    )
    scene.wait(hold_end)


def play_full_film(scene: Scene) -> None:
    """One-shot Acts 0–7 with review-friendly holds (for manim-full)."""
    stage = play_act1_stage(scene, hold_end=1.0)
    play_act2_project_k0(scene, stage, hold_end=1.0)
    play_act3_roll_k1(scene, stage, hold_end=0.9)
    play_act4_project_k1(scene, stage, hold_end=1.1)
    play_act5_roll_k2(scene, stage, hold_end=0.9)
    selected = play_act6a_select_k2(scene, stage, hold_end=0.7)
    play_act6b_fresh_b0(scene, stage, selected, hold_end=0.9)
    play_act6c_reentry_b1(scene, stage, selected, hold_end=1.0)
    play_act7_reveal_output(scene, stage, hold_end=1.8)


class Act7RevealOutput(Scene):
    """Act 7: fade input / dib_diffs / k; gold-flash center output."""

    def construct(self) -> None:
        stage = play_act1_stage(self, hold_end=0.1)
        play_act2_project_k0(self, stage, hold_end=0.15)
        play_act3_roll_k1(self, stage, hold_end=0.18)
        play_act4_project_k1(self, stage, hold_end=0.18)
        play_act5_roll_k2(self, stage, hold_end=0.2)
        selected = play_act6a_select_k2(self, stage, hold_end=0.2)
        play_act6b_fresh_b0(self, stage, selected, hold_end=0.2)
        play_act6c_reentry_b1(self, stage, selected, hold_end=0.35)
        play_act7_reveal_output(self, stage, hold_end=1.8)


class LpapFullFilm(Scene):
    """Complete pedagogical film through Act 7 — render once, watch once."""

    def construct(self) -> None:
        play_full_film(self)
