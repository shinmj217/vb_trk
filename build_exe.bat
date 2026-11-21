@echo off
rem Build the viewer-only exe: build_exe.bat  ->  vb_viewer.exe (keep it in the project folder)
rem Installs PyInstaller and the light viewer packages into an isolated venv (.venv-pack).
rem torch / SAM3 / ultralytics are NOT included: analysis still needs the project's Python environment.
cd /d "%~dp0"
if not exist .venv-pack python -m venv .venv-pack
if errorlevel 1 exit /b 1
.venv-pack\Scripts\python -m pip install -q pyinstaller fastapi uvicorn pandas numpy opencv-python-headless
if errorlevel 1 exit /b 1
.venv-pack\Scripts\pyinstaller --noconfirm --onefile --name vb_viewer --add-data "ui;ui" --collect-submodules uvicorn app.py
if errorlevel 1 exit /b 1
copy /Y dist\vb_viewer.exe vb_viewer.exe
echo.
echo Done: vb_viewer.exe (double-click to open the viewer)
