# Copyright 2026 Jigyasu Juneja
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Set-of-Marks (SoM) Discrete Visual UI Next-Hop Navigator."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .types import MAX_OPTIONS_PER_SLOT, CandidateMark, DecisionQuestion

GRID_ROWS = ("top", "middle", "bottom")
GRID_COLS = ("left", "center", "right")
GRID_4X4_ROWS = ("r0", "r1", "r2", "r3")
GRID_4X4_COLS = ("c0", "c1", "c2", "c3")


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ih = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iw = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ih * iw
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def point_in_box(cy: float, cx: float, box: tuple[float, float, float, float]) -> bool:
    return box[0] <= cy <= box[2] and box[1] <= cx <= box[3]


def assign_point_to_grid_cell(cy: float, cx: float) -> str:
    r = max(0, min(2, int((cy / 100.0) * 3.0)))
    c = max(0, min(2, int((cx / 100.0) * 3.0)))
    return f"{GRID_ROWS[r]}_{GRID_COLS[c]}"


def assign_box_to_grid_cell(box: tuple[float, float, float, float]) -> str:
    cy, cx = box_center(box)
    return assign_point_to_grid_cell(cy, cx)


def assign_point_to_grid_4x4(cy: float, cx: float) -> str:
    """Map normalized (cy, cx) in [0, 100] to a 4x4 spatial cell ('r0_c0' .. 'r3_c3')."""
    r = max(0, min(3, int((cy / 100.0) * 4.0)))
    c = max(0, min(3, int((cx / 100.0) * 4.0)))
    return f"{GRID_4X4_ROWS[r]}_{GRID_4X4_COLS[c]}"


def assign_box_to_grid_4x4(box: tuple[float, float, float, float]) -> str:
    """Map a [y_min, x_min, y_max, x_max] box center to its 4x4 spatial cell."""
    cy, cx = box_center(box)
    return assign_point_to_grid_4x4(cy, cx)


def grid_4x4_cell_bounds(cell: str) -> tuple[float, float, float, float]:
    """Return [y_min, x_min, y_max, x_max] bounds in [0, 100] for a 4x4 cell."""
    parts = cell.strip().split("_")
    if len(parts) != 2 or parts[0] not in GRID_4X4_ROWS or parts[1] not in GRID_4X4_COLS:
        raise ValueError(f"invalid 4x4 grid cell: {cell!r}")
    r = GRID_4X4_ROWS.index(parts[0])
    c = GRID_4X4_COLS.index(parts[1])
    return (r * 25.0, c * 25.0, (r + 1) * 25.0, (c + 1) * 25.0)


def expected_mark_center_from_probabilities(
    probs: Mapping[str, float],
    marks: Sequence[CandidateMark],
    catch_all_key: str = "none_visible",
) -> tuple[float, float, bool]:
    """Compute the probability-weighted expected (cy, cx) across candidate Set-of-Marks boxes.

    Excludes catch_all_key mass and renormalizes over valid mark IDs. Returns (0.0, 0.0, False)
    when no positive probability mass falls on candidate marks.
    """
    mass = 0.0
    sum_y = 0.0
    sum_x = 0.0
    for m in marks:
        mid = m.id.strip()
        if not mid or mid == catch_all_key:
            continue
        p = float(probs.get(mid, 0.0))
        if math.isfinite(p) and p > 0.0:
            mcy, mcx = box_center(m.box)
            mass += p
            sum_y += p * mcy
            sum_x += p * mcx
    if mass <= 1e-12:
        return 0.0, 0.0, False
    return sum_y / mass, sum_x / mass, True


def safe_expected_mark_center(
    probs: Mapping[str, float],
    marks: Sequence[CandidateMark],
    catch_all_key: str = "none_visible",
    max_dispersion: float = 18.0,
) -> tuple[float, float, CandidateMark | None, bool]:
    """Unimodal-safe expected Set-of-Marks center with automatic argmax snapping on bimodal/non-convex splits.

    When probability mass splits across spatially distant marks (spatial std dev > max_dispersion) or
    the weighted centroid lands in dead space outside every candidate mark box, snaps deterministically
    to the highest-probability candidate mark's center so clicks never land between buttons.
    """
    weighted: list[tuple[float, float, float, CandidateMark]] = []
    mass = 0.0
    for m in marks:
        mid = m.id.strip()
        if not mid or mid == catch_all_key:
            continue
        p = float(probs.get(mid, 0.0))
        if math.isfinite(p) and p > 0.0:
            mcy, mcx = box_center(m.box)
            weighted.append((p, mcy, mcx, m))
            mass += p

    if mass <= 1e-12 or not weighted:
        return 0.0, 0.0, None, False

    ey = sum(p * y for p, y, _, _ in weighted) / mass
    ex = sum(p * x for p, _, x, _ in weighted) / mass
    variance = sum((p / mass) * ((y - ey) ** 2 + (x - ex) ** 2) for p, y, x, _ in weighted)
    dispersion = math.sqrt(max(0.0, variance))

    containing, dist = select_containing_or_nearest_mark(ey, ex, [m for _, _, _, m in weighted])
    if dispersion <= max_dispersion and dist == 0.0 and containing is not None:
        return ey, ex, containing, True

    # Bimodal split or non-convex gap: snap to highest-probability candidate mark
    best_p, best_y, best_x, best_mark = max(weighted, key=lambda t: (t[0], -box_area(t[3].box)))
    _ = best_p
    return best_y, best_x, best_mark, True


def _cell_rc(cell: str) -> tuple[int, int] | None:
    parts = cell.split("_")
    if len(parts) != 2 or parts[0] not in GRID_ROWS or parts[1] not in GRID_COLS:
        return None
    return GRID_ROWS.index(parts[0]), GRID_COLS.index(parts[1])


def filter_marks_by_grid_cell(
    cell: str,
    marks: Sequence[CandidateMark],
    include_adjacent: bool = False,
) -> list[CandidateMark]:
    rc = _cell_rc(cell)
    if rc is None:
        return []
    tr, tc = rc
    out: list[CandidateMark] = []
    for m in marks:
        mrc = _cell_rc(assign_box_to_grid_cell(m.box))
        if mrc is None:
            continue
        r, c = mrc
        if (r, c) == (tr, tc) or (include_adjacent and abs(r - tr) + abs(c - tc) <= 1):
            out.append(m)
    return out


def select_containing_or_nearest_mark(
    cy: float,
    cx: float,
    marks: Sequence[CandidateMark],
) -> tuple[CandidateMark | None, float]:
    """Select the smallest-area mark containing (cy, cx), or the nearest mark by center distance."""
    if not marks:
        return None, float("inf")
    containing: list[tuple[float, float, int, CandidateMark]] = []
    nearest: list[tuple[float, int, CandidateMark]] = []
    for idx, m in enumerate(marks):
        mcy, mcx = box_center(m.box)
        dist = math.hypot(cy - mcy, cx - mcx)
        nearest.append((dist, idx, m))
        if point_in_box(cy, cx, m.box):
            containing.append((box_area(m.box), dist, idx, m))
    if containing:
        containing.sort(key=lambda t: (t[0], t[1], t[2]))
        return containing[0][3], 0.0
    nearest.sort(key=lambda t: (t[0], t[1]))
    return nearest[0][2], nearest[0][0]


def build_set_of_marks_questions(
    goal: str,
    marks: Sequence[CandidateMark],
) -> dict[str, DecisionQuestion]:
    """Construct the 4-slot single-pass Set-of-Marks UI navigation question map."""
    goal = goal.strip()
    if not goal:
        raise ValueError("goal must not be empty")
    if not marks:
        raise ValueError("marks must not be empty")
    if len(marks) > MAX_OPTIONS_PER_SLOT - 1:
        raise ValueError(
            f"marks count {len(marks)} exceeds single-pass Set-of-Marks limit {MAX_OPTIONS_PER_SLOT - 1}"
        )

    mark_criteria: dict[str, str] = {}
    mark_order: list[str] = []
    for m in marks:
        mid = m.id.strip()
        if not mid or mid == "none_visible":
            raise ValueError(f"invalid or reserved candidate mark id: {mid!r}")
        if mid in mark_criteria:
            raise ValueError(f"duplicate candidate mark id: {mid!r}")
        cell = assign_box_to_grid_cell(m.box)
        label = m.label.strip() or "interactive UI element"
        mark_criteria[mid] = (
            f"{label} (grid_cell={cell}, box=[{m.box[0]:.1f}, {m.box[1]:.1f}, {m.box[2]:.1f}, {m.box[3]:.1f}])"
        )
        mark_order.append(mid)

    mark_criteria["none_visible"] = (
        "Target UI control is off-screen, occluded, or requires scrolling before clicking"
    )
    mark_order.append("none_visible")

    return {
        "next_target_mark": DecisionQuestion(
            type="choice",
            instructions=f"Which numbered Set-of-Marks (SoM) UI element should be activated next to accomplish the goal: {goal}?",
            criteria=mark_criteria,
            order=mark_order,
        ),
        "element_state": DecisionQuestion(
            type="choice",
            instructions="What is the interactive state of the selected target UI element?",
            criteria={
                "interactive_ready": "Visible, enabled, and ready for immediate click or tap",
                "disabled_grayed_out": "Visible but disabled or awaiting prerequisite form input",
                "occluded_by_modal": "Partially or fully blocked by a modal, toast, or cookie banner",
                "requires_scroll": "Not visible in the current viewport; scrolling is required",
            },
            order=["interactive_ready", "disabled_grayed_out", "occluded_by_modal", "requires_scroll"],
        ),
        "loop_or_error_detected": DecisionQuestion(
            type="choice",
            instructions="Is the screen stuck in a validation error, modal trap, or repeated state?",
            criteria={
                "no": "Screen state is normal with no blocking error or loop",
                "yes": "Inline validation error, blocking dialog, or repeated state detected",
            },
            order=["no", "yes"],
        ),
        "goal_progress_score": DecisionQuestion(
            type="score",
            instructions="How close is the current screen state to completing the user goal?",
            levels=[
                "blocked_or_regressed",
                "initial_screen",
                "intermediate_step_complete",
                "final_confirmation_visible",
            ],
        ),
    }
