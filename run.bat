@echo off
REM MedSeg - installation (first run only) and launch, Windows.
REM Double-click this file, or run it from a terminal in the project folder.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [MedSeg] Creation de l'environnement virtuel .venv ...
    py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
)
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [MedSeg] Python introuvable. Installez Python 3.11 ou 3.12 depuis https://www.python.org
    pause
    exit /b 1
)

if not exist ".venv\.medseg_installed" (
    echo [MedSeg] Installation des dependances ^(premiere execution, quelques minutes^) ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [MedSeg] Echec de l'installation des dependances.
        pause
        exit /b 1
    )
    "%PY%" -m pip install -r requirements-dl.txt
    if errorlevel 1 echo [MedSeg] TensorFlow non installe : la partie deep learning sera desactivee.
    echo ok> ".venv\.medseg_installed"
)

echo [MedSeg] Lancement : http://localhost:8501
"%PY%" -m streamlit run app\streamlit_app.py
pause
