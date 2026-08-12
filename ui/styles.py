APP_STYLESHEET = """
QWidget {
    color: #172033;
    font-family: "Segoe UI";
    font-size: 13px;
}
QMainWindow, QWidget#appRoot, QWidget#loginRoot {
    background: #f4f7fb;
}
QDialog, QMessageBox, QInputDialog {
    background: #f4f7fb;
}
QMessageBox QLabel, QInputDialog QLabel {
    background: transparent;
    color: #172033;
}
QWidget#cameraSettingsViewport, QWidget#cameraSettingsContent {
    background: #f4f7fb;
}
QFrame#loginCard, QFrame#contentCard, QFrame#cameraCard, QGroupBox {
    background: white;
    border: 1px solid #dce3ee;
    border-radius: 10px;
}
QFrame#recordPhotoFrame {
    background: #e8edf5;
    border: 1px solid #d5deeb;
    border-radius: 10px;
}
QLabel#recordPhotoLabel {
    background: transparent;
    color: #6d7890;
    font-size: 14px;
}
QLabel#appTitle {
    color: #14213d;
    font-size: 24px;
    font-weight: 700;
}
QLabel#pageTitle {
    color: #14213d;
    font-size: 20px;
    font-weight: 700;
}
QLabel#sectionTitle {
    color: #26344f;
    font-size: 16px;
    font-weight: 600;
}
QLabel#mutedLabel {
    color: #6d7890;
}
QLineEdit, QComboBox {
    background: white;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px;
    min-height: 20px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #3468d4;
}
QComboBox QAbstractItemView {
    background: white;
    color: #172033;
    border: 1px solid #cbd5e1;
    selection-background-color: #3468d4;
    selection-color: white;
    outline: 0;
}
QPushButton {
    background: #2f66d0;
    color: white;
    border: 0;
    border-radius: 6px;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #2859b7; }
QPushButton:pressed { background: #214b9b; }
QPushButton#secondaryButton, QPushButton#openCaptureFileButton {
    background: #e8edf5;
    color: #25334d;
}
QPushButton#navButton {
    background: transparent;
    color: #cfdaee;
    text-align: left;
    padding: 11px 16px;
    font-weight: 500;
}
QPushButton#navButton:hover { background: #263a62; }
QPushButton#navButton:checked {
    background: #3468d4;
    color: white;
}
QFrame#sidebar { background: #182743; }
QLabel#sidebarTitle {
    color: white;
    font-size: 17px;
    font-weight: 700;
}
QHeaderView::section {
    background: #edf1f7;
    color: #37445d;
    border: 0;
    border-bottom: 1px solid #d7deea;
    padding: 8px;
    font-weight: 600;
}
QTableWidget {
    background: white;
    border: 1px solid #dce3ee;
    border-radius: 7px;
    gridline-color: #edf0f5;
    alternate-background-color: #f8fafc;
}
QGroupBox {
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
}
"""
