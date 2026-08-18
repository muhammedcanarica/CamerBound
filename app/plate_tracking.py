from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

from app.plate_detector import PlateDetection


LOGGER = logging.getLogger(__name__)
TRACK_LIMIT_LOG_INTERVAL_SECONDS = 5.0
TRACK_CANDIDATE_LIMIT = 8
TRACK_CENTER_DISTANCE_THRESHOLD = 1.25


@dataclass(slots=True)
class _TrackCandidateEvidence:
    observations: int = 0
    confidence_sum: float = 0.0
    best_confidence: float = 0.0

    def add(self, confidence: float) -> None:
        bounded = max(0.0, min(1.0, float(confidence)))
        self.observations += 1
        self.confidence_sum += bounded
        self.best_confidence = max(self.best_confidence, bounded)

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / max(1, self.observations)


@dataclass(slots=True)
class PlateTrack:
    track_id: int
    camera_id: int
    bbox: tuple[int, int, int, int]
    created_at: float
    first_seen_at: float
    last_seen_at: float
    last_activity_at: float
    last_detection_confidence: float
    pending_ocr_count: int = 0
    last_ocr_at: float | None = None
    best_text: str | None = None
    best_confidence: float = 0.0
    finalized: bool = False
    retired_at: float | None = None
    ocr_candidates: dict[str, _TrackCandidateEvidence] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlateTrackSnapshot:
    track_id: int
    camera_id: int
    bbox: tuple[int, int, int, int]
    created_at: float
    first_seen_at: float
    last_seen_at: float
    last_detection_confidence: float
    pending_ocr_count: int
    best_text: str | None
    best_confidence: float
    finalized: bool


@dataclass(frozen=True, slots=True)
class PlateTrackAssignment:
    detection: PlateDetection
    track_id: int
    iou: float
    center_distance_ratio: float
    created: bool = False


@dataclass(frozen=True, slots=True)
class PlateTrackingUpdate:
    assignments: tuple[PlateTrackAssignment, ...]
    active_tracks: tuple[PlateTrackSnapshot, ...]
    ignored_detections: tuple[PlateDetection, ...] = ()


class PlateTrackManager:
    """Small camera-scoped bbox tracker shared by detector and OCR threads."""

    def __init__(
        self,
        *,
        max_active_tracks_per_camera: int,
        timeout_ms: int,
        iou_threshold: float,
    ) -> None:
        self.max_active_tracks_per_camera = max(1, int(max_active_tracks_per_camera))
        self.timeout_seconds = max(0.1, int(timeout_ms) / 1000.0)
        self.iou_threshold = max(0.0, min(1.0, float(iou_threshold)))
        self._active: dict[int, dict[int, PlateTrack]] = {}
        self._tracks_by_id: dict[int, PlateTrack] = {}
        self._retired: dict[int, PlateTrack] = {}
        self._finalized_ids: deque[int] = deque()
        self._finalization_queued: set[int] = set()
        self._last_limit_log_at: dict[int, float] = {}
        self._next_track_id = 1
        self._lock = threading.Lock()

    def update(
        self,
        camera_id: int,
        detections: Sequence[PlateDetection],
        observed_at: float,
        *,
        activity_at: float | None = None,
    ) -> PlateTrackingUpdate:
        lifecycle_at = observed_at if activity_at is None else activity_at
        with self._lock:
            self._expire_due_locked(lifecycle_at)
            active = self._active.setdefault(camera_id, {})
            tracks = tuple(active.values())
            pairs: list[tuple[tuple[int, float, float], int, int, float, float]] = []
            for detection_index, detection in enumerate(detections):
                for track in tracks:
                    iou = bbox_iou(_detection_bbox(detection), track.bbox)
                    distance = normalized_center_distance(
                        _detection_bbox(detection),
                        track.bbox,
                    )
                    if (
                        iou < self.iou_threshold
                        and distance > TRACK_CENTER_DISTANCE_THRESHOLD
                    ):
                        continue
                    score = (
                        int(iou >= self.iou_threshold),
                        iou,
                        -distance,
                    )
                    pairs.append(
                        (score, detection_index, track.track_id, iou, distance)
                    )

            used_detections: set[int] = set()
            used_tracks: set[int] = set()
            assignments: list[PlateTrackAssignment] = []
            for _score, detection_index, track_id, iou, distance in sorted(
                pairs,
                key=lambda item: item[0],
                reverse=True,
            ):
                if detection_index in used_detections or track_id in used_tracks:
                    continue
                detection = detections[detection_index]
                track = active[track_id]
                self._refresh_track(
                    track,
                    detection,
                    observed_at,
                    lifecycle_at,
                )
                used_detections.add(detection_index)
                used_tracks.add(track_id)
                assignments.append(
                    PlateTrackAssignment(detection, track_id, iou, distance)
                )
                LOGGER.debug(
                    "Plate detection matched to track camera_id=%s track_id=%s "
                    "iou=%.3f center_distance_ratio=%.3f bbox=%s",
                    camera_id,
                    track_id,
                    iou,
                    distance,
                    _detection_bbox(detection),
                )

            ignored: list[PlateDetection] = []
            for detection_index, detection in enumerate(detections):
                if detection_index in used_detections:
                    continue
                if len(active) >= self.max_active_tracks_per_camera:
                    ignored.append(detection)
                    self._log_track_limit(camera_id)
                    continue
                track = self._create_track(
                    camera_id,
                    detection,
                    observed_at,
                    lifecycle_at,
                )
                active[track.track_id] = track
                self._tracks_by_id[track.track_id] = track
                assignments.append(
                    PlateTrackAssignment(
                        detection,
                        track.track_id,
                        0.0,
                        math.inf,
                        created=True,
                    )
                )
                LOGGER.debug(
                    "Plate track created camera_id=%s track_id=%s bbox=%s",
                    camera_id,
                    track.track_id,
                    track.bbox,
                )

            assignments.sort(
                key=lambda item: detections.index(item.detection)
            )
            return PlateTrackingUpdate(
                assignments=tuple(assignments),
                active_tracks=self._camera_snapshots_locked(camera_id),
                ignored_detections=tuple(ignored),
            )

    def expire_due(self, now: float | None = None) -> None:
        with self._lock:
            self._expire_due_locked(time.monotonic() if now is None else now)

    def next_expiration(self) -> float | None:
        with self._lock:
            deadlines = [
                track.last_activity_at + self.timeout_seconds
                for tracks in self._active.values()
                for track in tracks.values()
            ]
            return min(deadlines, default=None)

    def mark_ocr_scheduled(self, track_id: int | None) -> bool:
        if track_id is None:
            return True
        with self._lock:
            track = self._tracks_by_id.get(track_id)
            if track is None or track.finalized:
                return False
            track.pending_ocr_count += 1
            LOGGER.debug(
                "OCR job scheduled for track camera_id=%s track_id=%s pending=%s",
                track.camera_id,
                track.track_id,
                track.pending_ocr_count,
            )
            return True

    def can_accept_ocr_result(self, track_id: int | None) -> bool:
        if track_id is None:
            return True
        with self._lock:
            track = self._tracks_by_id.get(track_id)
            return bool(
                track is not None
                and not track.finalized
                and track.pending_ocr_count > 0
            )

    def record_ocr_result(
        self,
        track_id: int | None,
        text: str,
        confidence: float,
        observed_at: float,
    ) -> bool:
        if track_id is None:
            return True
        with self._lock:
            track = self._tracks_by_id.get(track_id)
            if track is None or track.finalized or track.pending_ocr_count <= 0:
                return False
            evidence = track.ocr_candidates.setdefault(text, _TrackCandidateEvidence())
            evidence.add(confidence)
            track.last_ocr_at = max(track.last_ocr_at or observed_at, observed_at)
            if len(track.ocr_candidates) > TRACK_CANDIDATE_LIMIT:
                weakest = min(
                    track.ocr_candidates,
                    key=lambda value: self._candidate_score(
                        value,
                        track.ocr_candidates[value],
                    ),
                )
                track.ocr_candidates.pop(weakest, None)
            if track.ocr_candidates:
                best_text = max(
                    track.ocr_candidates,
                    key=lambda value: self._candidate_score(
                        value,
                        track.ocr_candidates[value],
                    ),
                )
                best = track.ocr_candidates[best_text]
                track.best_text = best_text
                track.best_confidence = best.best_confidence
            LOGGER.debug(
                "OCR result received for track camera_id=%s track_id=%s "
                "text=%s confidence=%.3f best=%s best_confidence=%.3f",
                track.camera_id,
                track.track_id,
                text,
                confidence,
                track.best_text or "none",
                track.best_confidence,
            )
            return True

    def mark_ocr_finished(self, track_id: int | None) -> None:
        if track_id is None:
            return
        with self._lock:
            track = self._tracks_by_id.get(track_id)
            if track is None:
                return
            track.pending_ocr_count = max(0, track.pending_ocr_count - 1)
            if track.retired_at is not None and track.pending_ocr_count == 0:
                self._queue_finalization_locked(track)

    def active_snapshots(self, camera_id: int) -> tuple[PlateTrackSnapshot, ...]:
        with self._lock:
            return self._camera_snapshots_locked(camera_id)

    def consume_finalized(self) -> tuple[PlateTrackSnapshot, ...]:
        with self._lock:
            snapshots: list[PlateTrackSnapshot] = []
            while self._finalized_ids:
                track_id = self._finalized_ids.popleft()
                self._finalization_queued.discard(track_id)
                track = self._retired.pop(track_id, None)
                if track is None:
                    continue
                snapshots.append(self._snapshot(track))
                self._tracks_by_id.pop(track_id, None)
            return tuple(snapshots)

    def clear(self) -> None:
        with self._lock:
            self._active.clear()
            self._tracks_by_id.clear()
            self._retired.clear()
            self._finalized_ids.clear()
            self._finalization_queued.clear()

    def _expire_due_locked(self, now: float) -> None:
        for camera_id, tracks in tuple(self._active.items()):
            for track_id, track in tuple(tracks.items()):
                if now - track.last_activity_at < self.timeout_seconds:
                    continue
                tracks.pop(track_id, None)
                track.retired_at = now
                self._retired[track_id] = track
                LOGGER.debug(
                    "Plate track expired camera_id=%s track_id=%s inactive_ms=%.1f "
                    "frame_age_ms=%.1f pending_ocr=%s",
                    camera_id,
                    track_id,
                    max(0.0, now - track.last_activity_at) * 1000.0,
                    max(0.0, now - track.last_seen_at) * 1000.0,
                    track.pending_ocr_count,
                )
                if track.pending_ocr_count == 0:
                    self._queue_finalization_locked(track)
            if not tracks:
                self._active.pop(camera_id, None)

    def _queue_finalization_locked(self, track: PlateTrack) -> None:
        if track.track_id in self._finalization_queued:
            return
        track.finalized = True
        self._finalization_queued.add(track.track_id)
        self._finalized_ids.append(track.track_id)
        LOGGER.debug(
            "Plate track finalized camera_id=%s track_id=%s best=%s "
            "best_confidence=%.3f",
            track.camera_id,
            track.track_id,
            track.best_text or "none",
            track.best_confidence,
        )

    def _create_track(
        self,
        camera_id: int,
        detection: PlateDetection,
        observed_at: float,
        activity_at: float,
    ) -> PlateTrack:
        track = PlateTrack(
            track_id=self._next_track_id,
            camera_id=camera_id,
            bbox=_detection_bbox(detection),
            created_at=observed_at,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
            last_activity_at=activity_at,
            last_detection_confidence=detection.confidence,
        )
        self._next_track_id += 1
        return track

    @staticmethod
    def _refresh_track(
        track: PlateTrack,
        detection: PlateDetection,
        observed_at: float,
        activity_at: float,
    ) -> None:
        if observed_at >= track.last_seen_at:
            track.bbox = _detection_bbox(detection)
            track.last_detection_confidence = detection.confidence
        track.last_seen_at = max(track.last_seen_at, observed_at)
        track.last_activity_at = max(track.last_activity_at, activity_at)

    def _camera_snapshots_locked(
        self,
        camera_id: int,
    ) -> tuple[PlateTrackSnapshot, ...]:
        return tuple(
            self._snapshot(track)
            for track in sorted(
                self._active.get(camera_id, {}).values(),
                key=lambda item: item.track_id,
            )
        )

    @staticmethod
    def _snapshot(track: PlateTrack) -> PlateTrackSnapshot:
        return PlateTrackSnapshot(
            track_id=track.track_id,
            camera_id=track.camera_id,
            bbox=track.bbox,
            created_at=track.created_at,
            first_seen_at=track.first_seen_at,
            last_seen_at=track.last_seen_at,
            last_detection_confidence=track.last_detection_confidence,
            pending_ocr_count=track.pending_ocr_count,
            best_text=track.best_text,
            best_confidence=track.best_confidence,
            finalized=track.finalized,
        )

    @staticmethod
    def _candidate_score(
        text: str,
        evidence: _TrackCandidateEvidence,
    ) -> tuple[int, float, float, str]:
        return (
            evidence.observations,
            evidence.average_confidence,
            evidence.best_confidence,
            text,
        )

    def _log_track_limit(self, camera_id: int) -> None:
        now = time.monotonic()
        previous = self._last_limit_log_at.get(camera_id)
        if previous is not None and now - previous < TRACK_LIMIT_LOG_INTERVAL_SECONDS:
            return
        self._last_limit_log_at[camera_id] = now
        LOGGER.debug(
            "Plate detection ignored because track limit reached camera_id=%s "
            "max_active_tracks=%s",
            camera_id,
            self.max_active_tracks_per_camera,
        )


def bbox_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    x1 = max(first_x, second_x)
    y1 = max(first_y, second_y)
    x2 = min(first_x + first_width, second_x + second_width)
    y2 = min(first_y + first_height, second_y + second_height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = first_width * first_height + second_width * second_height - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    first_center = (first_x + first_width / 2.0, first_y + first_height / 2.0)
    second_center = (
        second_x + second_width / 2.0,
        second_y + second_height / 2.0,
    )
    distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    scale = max(
        math.hypot(first_width, first_height),
        math.hypot(second_width, second_height),
        1.0,
    )
    return distance / scale


def _detection_bbox(detection: PlateDetection) -> tuple[int, int, int, int]:
    return detection.x, detection.y, detection.width, detection.height
