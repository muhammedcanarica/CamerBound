@echo off
setlocal EnableExtensions

cd /d "%~dp0.."
if errorlevel 1 (
    echo ERROR: Repository root could not be opened.
    exit /b 1
)

set "REPO_ROOT=%CD%"
set "PORTABLE_EXE=%REPO_ROOT%\dist\CamerBound\CamerBound.exe"
set "PORTABLE_DATA=%REPO_ROOT%\dist\CamerBound\data"
set "ISS_FILE=%REPO_ROOT%\packaging\installer\CamerBound.iss"
set "OUTPUT_DIR=%REPO_ROOT%\dist-installer"
set "OUTPUT_EXE=%OUTPUT_DIR%\CamerBound_Setup.exe"
set "ISCC_EXE="

if not exist "%PORTABLE_EXE%" (
    echo ERROR: Missing verified portable build: %PORTABLE_EXE%
    exit /b 1
)
if not exist "%ISS_FILE%" (
    echo ERROR: Missing Inno Setup script: %ISS_FILE%
    exit /b 1
)

powershell -NoProfile -Command "if ((Get-ChildItem -LiteralPath '%PORTABLE_DATA%' -Recurse -File -Force -ErrorAction Stop | Measure-Object).Count -ne 0) { throw 'Portable data directory is not clean.' }"
if errorlevel 1 exit /b 1

for %%I in (
    "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
    "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    "%ProgramFiles%\Inno Setup 6\ISCC.exe"
) do (
    if not defined ISCC_EXE if exist "%%~I" set "ISCC_EXE=%%~I"
)
if not defined ISCC_EXE (
    for /f "delims=" %%I in ('where ISCC.exe 2^>nul') do if not defined ISCC_EXE set "ISCC_EXE=%%~fI"
)
if not defined ISCC_EXE (
    echo ERROR: Inno Setup 6 ISCC.exe was not found.
    echo Install the official Inno Setup 6 package from https://jrsoftware.org/isdl.php
    exit /b 1
)

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
powershell -NoProfile -Command "& { param([string]$root, [string]$target) $root = [IO.Path]::GetFullPath($root).TrimEnd([IO.Path]::DirectorySeparatorChar); $target = [IO.Path]::GetFullPath($target); $expected = [IO.Path]::Combine($root, 'dist-installer', 'CamerBound_Setup.exe'); if (-not $target.Equals($expected, [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe installer cleanup target: ' + $target }; if (Test-Path -LiteralPath $target -PathType Leaf) { Remove-Item -LiteralPath $target -Force -ErrorAction Stop } }" "%REPO_ROOT%" "%OUTPUT_EXE%"
if errorlevel 1 exit /b 1

echo Inno Setup compiler: %ISCC_EXE%
"%ISCC_EXE%" /Qp "%ISS_FILE%"
if errorlevel 1 (
    echo ERROR: Inno Setup compilation failed.
    exit /b 1
)

if not exist "%OUTPUT_EXE%" (
    echo ERROR: Expected installer was not created.
    exit /b 1
)

set "INSTALLER_SHA256="
for /f "skip=1 delims=" %%H in ('certutil.exe -hashfile "%OUTPUT_EXE%" SHA256 2^>nul') do if not defined INSTALLER_SHA256 set "INSTALLER_SHA256=%%H"
set "INSTALLER_SHA256=%INSTALLER_SHA256: =%"
if not defined INSTALLER_SHA256 (
    echo ERROR: Installer SHA-256 could not be calculated.
    exit /b 1
)

echo INSTALLER BUILD SUCCESS
echo SHA-256: %INSTALLER_SHA256%
echo %OUTPUT_EXE%
exit /b 0
