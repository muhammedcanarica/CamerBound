from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH).resolve().parent

runtime_packages = (
    "bcrypt",
    "onnxruntime",
    "openvino",
    "paddle",
    "paddleocr",
    "paddlex",
)
runtime_distributions = (
    "bcrypt",
    "imagesize",
    "onnxruntime",
    "openvino",
    "opencv-contrib-python",
    "paddleocr",
    "paddlepaddle",
    "paddlex",
    "pyclipper",
    "pypdfium2",
    "PySide6",
    "python-bidi",
    "shapely",
)

datas = []
binaries = []
hiddenimports = []

for package in runtime_packages:
    datas += collect_data_files(package)
    binaries += collect_dynamic_libs(package)
    hiddenimports += collect_submodules(package, on_error="ignore")

for distribution in runtime_distributions:
    try:
        datas += copy_metadata(distribution)
    except Exception:
        # Runtime imports are authoritative; metadata is needed only by packages
        # that query their installed distribution version.
        pass

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib",
        "pytest",
        "tensorflow",
        "tkinter",
        "torch",
        "torchvision",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CamerBound",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CamerBound",
)
