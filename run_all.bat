@echo off
REM Railway-TI-Guardian - one-click setup and run (Windows)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Railway-TI-Guardian...
python main.py

pause
