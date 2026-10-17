#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python -c 'import streamlit; from api import health; assert health()["model_loaded"]'
