from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Component:
    component_id: int
    area: int
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict:
        return asdict(self)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def extract_components(
    mask: np.ndarray,
    min_area: int = 5,
    connectivity: int = 2,
) -> list[Component]:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    structure = ndimage.generate_binary_structure(2, connectivity)
    labeled, count = ndimage.label(mask, structure=structure)
    objects = ndimage.find_objects(labeled)
    components: list[Component] = []
    for component_id in range(1, count + 1):
        slc = objects[component_id - 1]
        if slc is None:
            continue
        local = labeled[slc] == component_id
        area = int(local.sum())
        if area < min_area:
            continue
        ys, xs = np.where(labeled == component_id)
        x1, x2 = float(xs.min()), float(xs.max() + 1)
        y1, y2 = float(ys.min()), float(ys.max() + 1)
        components.append(
            Component(
                component_id=component_id,
                area=area,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                cx=(x1 + x2) / 2.0,
                cy=(y1 + y2) / 2.0,
            )
        )
    return sorted(components, key=lambda c: c.area, reverse=True)


def component_mask(mask: np.ndarray, component: Component) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, _ = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    return (labeled == component.component_id).astype(np.uint8)


def side_candidates(
    components: Iterable[Component], image_width: int
) -> dict[str, list[Component]]:
    result = {"left": [], "right": []}
    midline = image_width / 2.0
    for component in components:
        # Slices are stored as (Y, X): columns are canonical X. In RAS,
        # increasing X points anatomically right, so low columns are left kidney.
        side = "left" if component.cx < midline else "right"
        result[side].append(component)
    return result


def continuity_cost(
    candidate: Component,
    previous: Component,
    image_shape: tuple[int, int],
) -> float:
    height, width = image_shape
    diagonal = max((height**2 + width**2) ** 0.5, 1.0)
    center_distance = (
        (candidate.cx - previous.cx) ** 2 + (candidate.cy - previous.cy) ** 2
    ) ** 0.5 / diagonal
    overlap_cost = 1.0 - box_iou(candidate.box, previous.box)
    area_cost = abs(log(max(candidate.area, 1) / max(previous.area, 1)))
    return 2.0 * center_distance + 0.7 * overlap_cost + 0.15 * area_cost


def track_side(
    candidates_by_slice: list[list[Component]],
    image_shape: tuple[int, int],
    max_gap: int = 3,
) -> list[Component | None]:
    selected: list[Component | None] = []
    previous: Component | None = None
    previous_index = -10**9
    for index, candidates in enumerate(candidates_by_slice):
        if not candidates:
            selected.append(None)
            continue
        if previous is not None and index - previous_index <= max_gap:
            chosen = min(
                candidates,
                key=lambda candidate: continuity_cost(candidate, previous, image_shape),
            )
        else:
            chosen = max(candidates, key=lambda candidate: candidate.area)
        selected.append(chosen)
        previous = chosen
        previous_index = index
    return selected


def build_dual_tracks(
    masks: list[np.ndarray],
    min_area: int = 5,
    max_gap: int = 3,
) -> tuple[list[Component | None], list[Component | None], dict]:
    if not masks:
        return [], [], {}
    image_shape = masks[0].shape
    per_slice = [extract_components(mask, min_area=min_area) for mask in masks]
    left_candidates: list[list[Component]] = []
    right_candidates: list[list[Component]] = []
    for components in per_slice:
        sides = side_candidates(components, image_shape[1])
        left_candidates.append(sides["left"])
        right_candidates.append(sides["right"])
    left_track = track_side(left_candidates, image_shape, max_gap=max_gap)
    right_track = track_side(right_candidates, image_shape, max_gap=max_gap)

    def audit(track: list[Component | None], side: str) -> dict:
        jumps: list[float] = []
        prior: Component | None = None
        for current in track:
            if current is None:
                continue
            if prior is not None:
                jump = (
                    (current.cx - prior.cx) ** 2 + (current.cy - prior.cy) ** 2
                ) ** 0.5 / max(image_shape)
                jumps.append(float(jump))
            prior = current
        crossings = sum(
            1
            for component in track
            if component is not None
            and ((side == "left" and component.cx >= image_shape[1] / 2.0)
                 or (side == "right" and component.cx < image_shape[1] / 2.0))
        )
        return {
            "present_slices": int(sum(x is not None for x in track)),
            "crossing_count": int(crossings),
            "adjacent_jump_mean": float(np.mean(jumps)) if jumps else 0.0,
            "adjacent_jump_max": float(np.max(jumps)) if jumps else 0.0,
            "large_jump_count_20pct": int(sum(x > 0.20 for x in jumps)),
        }

    audit_summary = {
        "left": audit(left_track, "left"),
        "right": audit(right_track, "right"),
        "slices_with_more_than_two_components": int(sum(len(x) > 2 for x in per_slice)),
    }
    return left_track, right_track, audit_summary


def match_component_to_box(
    mask: np.ndarray,
    box: Sequence[float],
    min_area: int = 1,
) -> np.ndarray:
    components = extract_components(mask, min_area=min_area)
    if not components:
        return np.zeros_like(mask, dtype=np.uint8)
    x1, y1, x2, y2 = map(float, box)
    gt_cx, gt_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    height, width = mask.shape
    diagonal = max((height**2 + width**2) ** 0.5, 1.0)

    def score(component: Component) -> float:
        distance = (
            (component.cx - gt_cx) ** 2 + (component.cy - gt_cy) ** 2
        ) ** 0.5 / diagonal
        return box_iou(component.box, box) - 0.25 * distance

    chosen = max(components, key=score)
    return component_mask(mask, chosen)
