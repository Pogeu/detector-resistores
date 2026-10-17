"""Rastreamento leve por associação de caixas para vídeo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Box = tuple[float, float, float, float]


def intersection_over_union(first: Box, second: Box) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: str
    bbox: Box
    missed: int = 0


class IoUTracker:
    """Associa caixas consecutivas, tolerando pequenos desaparecimentos."""

    def __init__(self, *, iou_threshold: float = 0.25, max_missed: int = 3) -> None:
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold deve estar entre 0 e 1")
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: list[_Track] = []

    def update(self, boxes: Iterable[Box]) -> list[str]:
        current = list(boxes)
        assignments: list[str | None] = [None] * len(current)
        used_tracks: set[int] = set()
        candidates = sorted(
            [
                (
                intersection_over_union(box, track.bbox),
                box_index,
                track_index,
                )
                for box_index, box in enumerate(current)
                for track_index, track in enumerate(self._tracks)
                if intersection_over_union(box, track.bbox) >= self.iou_threshold
            ],
            reverse=True,
        )
        for overlap, box_index, track_index in candidates:
            if assignments[box_index] is not None or track_index in used_tracks:
                continue
            assignments[box_index] = self._tracks[track_index].track_id
            self._tracks[track_index].bbox = current[box_index]
            self._tracks[track_index].missed = 0
            used_tracks.add(track_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in used_tracks:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]

        for box_index, box in enumerate(current):
            if assignments[box_index] is None:
                track_id = f"track-{self._next_id}"
                self._next_id += 1
                self._tracks.append(_Track(track_id, box))
                assignments[box_index] = track_id
        return [assignment or "track-unknown" for assignment in assignments]
