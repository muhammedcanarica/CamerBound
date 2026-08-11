from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from app.config import NormalizedRoi


class RoiCanvas(QLabel):
    def __init__(self, frame: np.ndarray, roi: NormalizedRoi) -> None:
        super().__init__()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        self._image = QImage(
            rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
        ).copy()
        self._roi = roi
        self._start: QPoint | None = None
        self._selection = QRect()
        self.setMinimumSize(720, 420)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def selected_roi(self) -> NormalizedRoi:
        if self._selection.isValid():
            image_rect = self._image_rect()
            return NormalizedRoi(
                x=(self._selection.left() - image_rect.left()) / image_rect.width(),
                y=(self._selection.top() - image_rect.top()) / image_rect.height(),
                width=self._selection.width() / image_rect.width(),
                height=self._selection.height() / image_rect.height(),
            )
        return self._roi

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        image_rect = self._image_rect()
        painter.drawPixmap(image_rect, QPixmap.fromImage(self._image))
        selected = self._selection if self._selection.isValid() else self._roi_rect(image_rect)
        painter.setPen(QPen(Qt.GlobalColor.green, 3))
        painter.drawRect(selected)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self._start = self._clip(event.position().toPoint())
            self._selection = QRect(self._start, self._start)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._start is not None:
            self._selection = QRect(self._start, self._clip(event.position().toPoint())).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._start is not None and event.button() is Qt.MouseButton.LeftButton:
            self._selection = QRect(self._start, self._clip(event.position().toPoint())).normalized()
            self._start = None
            self.update()

    def _image_rect(self) -> QRect:
        scaled = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )

    def _roi_rect(self, image_rect: QRect) -> QRect:
        return QRect(
            image_rect.left() + round(self._roi.x * image_rect.width()),
            image_rect.top() + round(self._roi.y * image_rect.height()),
            max(1, round(self._roi.width * image_rect.width())),
            max(1, round(self._roi.height * image_rect.height())),
        )

    def _clip(self, point: QPoint) -> QPoint:
        rect = self._image_rect()
        return QPoint(
            max(rect.left(), min(rect.right(), point.x())),
            max(rect.top(), min(rect.bottom(), point.y())),
        )


class RoiCalibrationDialog(QDialog):
    def __init__(self, frame: np.ndarray, roi: NormalizedRoi, parent: object = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plaka ROI Kalibrasyonu")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Plakanın görüneceği alanı fareyle çizin."))
        self.canvas = RoiCanvas(frame, roi)
        layout.addWidget(self.canvas)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_roi(self) -> NormalizedRoi:
        return self.canvas.selected_roi()
