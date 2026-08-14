@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv-build\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Creating isolated ONNX build environment...
    python -m venv .venv-build || exit /b 1
    "%PYTHON%" -m pip install --upgrade pip || exit /b 1
    "%PYTHON%" -m pip install -r requirements.txt || exit /b 1
)

echo Building ONNX mosaic directory...
"%PYTHON%" -m PyInstaller mosaic.spec --noconfirm %*
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)
echo.
echo Build succeeded! dist\mosaic\mosaic.exe ready.

echo.
echo Building installer...
set "ISCC="
where ISCC.exe >nul 2>nul && set "ISCC=ISCC.exe"
if not defined ISCC if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not defined ISCC if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"

if not defined ISCC (
    echo Inno Setup ^(ISCC.exe^) not found - skipping installer build.
    echo   Install it with: winget install JRSoftware.InnoSetup
    goto :end
)

"%ISCC%" installer.iss
if %errorlevel% neq 0 (
    echo Installer build failed!
    exit /b 1
)
echo Installer ready: installer_output\MosaicSetup.exe

:end
endlocal
