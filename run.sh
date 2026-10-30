#!/usr/bin/env bash
# MedSeg - installation (first run only) and launch, Linux / macOS:  ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "[MedSeg] Création de l'environnement virtuel .venv ..."
    for py in python3.12 python3.11 python3.10 python3; do
        if command -v "$py" >/dev/null 2>&1; then "$py" -m venv .venv && break; fi
    done
fi
PY=.venv/bin/python
[ -x "$PY" ] || { echo "[MedSeg] Python 3.10+ introuvable."; exit 1; }

if [ ! -f .venv/.medseg_installed ]; then
    echo "[MedSeg] Installation des dépendances (première exécution, quelques minutes) ..."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
    "$PY" -m pip install -r requirements-dl.txt \
        || echo "[MedSeg] TensorFlow non installé : la partie deep learning sera désactivée."
    touch .venv/.medseg_installed
fi

echo "[MedSeg] Lancement : http://localhost:8501"
exec "$PY" -m streamlit run app/streamlit_app.py
