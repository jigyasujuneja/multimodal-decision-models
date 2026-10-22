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
from typing import Sequence

from .types import MAX_OPTIONS_PER_SLOT, CandidateMark, DecisionQuestion

GRID_ROWS = ("top", "middle", "bottom")
GRID_COLS = ("left", "center", "right")


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
