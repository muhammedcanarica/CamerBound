@echo off
setlocal EnableExtensions

cd /d "%~dp0.."
if errorlevel 1 (
    echo ERROR: Repository root could not be opened.
    exit /b 1
)

set "REPO_ROOT=%CD%"
set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
set "SPEC=%REPO_ROOT%\packaging\CamerBound.spec"
set "BUILD_DIR=%REPO_ROOT%\build"
set "DIST_DIR=%REPO_ROOT%\dist"
set "APP_DIR=%DIST_DIR%\CamerBound"
set "PYTHONNOUSERSITE=1"

if not exist "%PYTHON%" (
    echo ERROR: Missing virtual environment Python: %PYTHON%
    exit /b 1
)
if not exist "%SPEC%" (
    echo ERROR: Missing PyInstaller spec: %SPEC%
    exit /b 1
)

for %%F in (
    "models\ocr\paddle\detection\inference.json"
    "models\ocr\paddle\detection\inference.pdiparams"
    "models\ocr\paddle\detection\inference.yml"
    "models\ocr\paddle\recognition\inference.json"
    "models\ocr\paddle\recognition\inference.pdiparams"
    "models\ocr\paddle\recognition\inference.yml"
    "models\plate_detector\vehicle-license-plate-detection-barrier-0123\model.xml"
    "models\plate_detector\vehicle-license-plate-detection-barrier-0123\model.bin"
) do (
    if not exist "%REPO_ROOT%\%%~F" (
        echo ERROR: Missing runtime model file: %%~F
        exit /b 1
    )
)

"%PYTHON%" -c "import PyInstaller, PySide6, cv2, bcrypt, paddle, paddleocr, onnxruntime, openvino; print('PyInstaller ' + PyInstaller.__version__ + ' - dependency check OK')"
if errorlevel 1 exit /b 1

call :remove_build_directory "%BUILD_DIR%" build
if errorlevel 1 exit /b 1
call :remove_build_directory "%DIST_DIR%" dist
if errorlevel 1 exit /b 1

"%PYTHON%" -m PyInstaller --noconfirm --clean --workpath "%BUILD_DIR%" --distpath "%DIST_DIR%" "%SPEC%"
if errorlevel 1 (
    echo ERROR: PyInstaller build failed.
    exit /b 1
)

if not exist "%APP_DIR%" mkdir "%APP_DIR%"
if not exist "%APP_DIR%\config" mkdir "%APP_DIR%\config"
if not exist "%APP_DIR%\data" mkdir "%APP_DIR%\data"
if not exist "%APP_DIR%\models\ocr\paddle\detection" mkdir "%APP_DIR%\models\ocr\paddle\detection"
if not exist "%APP_DIR%\models\ocr\paddle\recognition" mkdir "%APP_DIR%\models\ocr\paddle\recognition"
if not exist "%APP_DIR%\models\plate_detector\vehicle-license-plate-detection-barrier-0123" mkdir "%APP_DIR%\models\plate_detector\vehicle-license-plate-detection-barrier-0123"

copy /Y "%REPO_ROOT%\config\settings.json" "%APP_DIR%\config\settings.json" >nul
if errorlevel 1 exit /b 1

for %%F in (inference.json inference.pdiparams inference.yml) do (
    copy /Y "%REPO_ROOT%\models\ocr\paddle\detection\%%F" "%APP_DIR%\models\ocr\paddle\detection\%%F" >nul
    if errorlevel 1 exit /b 1
    copy /Y "%REPO_ROOT%\models\ocr\paddle\recognition\%%F" "%APP_DIR%\models\ocr\paddle\recognition\%%F" >nul
    if errorlevel 1 exit /b 1
)
if exist "%REPO_ROOT%\models\ocr\paddle\model-info.json" (
    copy /Y "%REPO_ROOT%\models\ocr\paddle\model-info.json" "%APP_DIR%\models\ocr\paddle\model-info.json" >nul
    if errorlevel 1 exit /b 1
)
copy /Y "%REPO_ROOT%\models\plate_detector\vehicle-license-plate-detection-barrier-0123\model.xml" "%APP_DIR%\models\plate_detector\vehicle-license-plate-detection-barrier-0123\model.xml" >nul
if errorlevel 1 exit /b 1
copy /Y "%REPO_ROOT%\models\plate_detector\vehicle-license-plate-detection-barrier-0123\model.bin" "%APP_DIR%\models\plate_detector\vehicle-license-plate-detection-barrier-0123\model.bin" >nul
if errorlevel 1 exit /b 1

if not exist "%APP_DIR%\CamerBound.exe" (
    echo ERROR: Expected executable was not created.
    exit /b 1
)

echo BUILD SUCCESS
echo %APP_DIR%\CamerBound.exe
exit /b 0

:remove_build_directory
set "REMOVE_TARGET=%~1"
set "EXPECTED_LEAF=%~2"
powershell -NoProfile -Command "& { param([string]$root, [string]$target, [string]$leaf) $root = [IO.Path]::GetFullPath($root).TrimEnd([IO.Path]::DirectorySeparatorChar); $target = [IO.Path]::GetFullPath($target).TrimEnd([IO.Path]::DirectorySeparatorChar); $expected = [IO.Path]::Combine($root, $leaf); if (-not $target.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe build cleanup target: ' + $target }; if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop } }" "%REPO_ROOT%" "%REMOVE_TARGET%" "%EXPECTED_LEAF%"
exit /b %ERRORLEVEL%
