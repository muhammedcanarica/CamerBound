from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera import CameraService, Direction
from app.config import load_config
from app.database import Database
from app.plate_recognition import PaddleOcrProvider, PlateRecognitionProcessor
from app.plate_service import PlateService


def iter_frames(path: Path, sample_every: int):
    if path.is_dir():
        image_paths = sorted(
            item for item in path.iterdir() if item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for index, image_path in enumerate(image_paths):
            if index % sample_every == 0:
                frame = cv2.imread(str(image_path))
                if frame is not None:
                    yield frame
        return

    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % sample_every == 0:
                yield frame
            index += 1
    finally:
        capture.release()


class TimedProvider:
    def __init__(self, inner: PaddleOcrProvider) -> None:
        self.inner = inner
        self.calls = 0
        self.total_ms = 0.0

    def recognize(self, images):
        started = time.perf_counter()
        try:
            return self.inner.recognize(images)
        finally:
            self.calls += 1
            self.total_ms += (time.perf_counter() - started) * 1000


def main() -> int:
    parser = argparse.ArgumentParser(description="Video veya görsel klasöründe headless OCR pipeline testi.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--direction", choices=("ENTRY", "EXIT"), default="ENTRY")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f"Girdi bulunamadı: {args.input}")
    if args.sample_every < 1:
        parser.error("--sample-every en az 1 olmalıdır.")

    config = load_config().plate_recognition
    provider = TimedProvider(PaddleOcrProvider(config.model_root))
    frame_count = candidates = confirmed = duplicates = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="camerbound-pipeline-") as temporary:
        database = Database(Path(temporary) / "pipeline.db")
        database.initialize()
        camera_service = CameraService(database)
        camera = next(
            item for item in camera_service.list_cameras() if item.direction is Direction(args.direction)
        )
        processor = PlateRecognitionProcessor(
            provider,
            PlateService(database, config.duplicate_cooldown_seconds),
            config,
        )
        for frame in iter_frames(args.input.resolve(), args.sample_every):
            outcome = processor.process(camera.id, camera.direction, frame)
            frame_count += 1
            candidates += int(outcome.candidate is not None)
            confirmed += int(outcome.record is not None)
            duplicates += int(outcome.duplicate)
            if args.max_frames and frame_count >= args.max_frames:
                break

    elapsed = time.perf_counter() - started
    average_ms = provider.total_ms / provider.calls if provider.calls else 0.0
    print(f"frames={frame_count} ocr_attempts={provider.calls} candidates={candidates}")
    print(f"confirmed={confirmed} duplicates={duplicates}")
    print(f"elapsed_seconds={elapsed:.3f} average_inference_ms={average_ms:.1f}")
    return 0 if frame_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
