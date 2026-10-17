from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_price_finder_loads_without_training():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_ui.py"
    with patch("sklearn.model_selection.cross_val_predict", side_effect=AssertionError("Unexpected training")):
        app = AppTest.from_file(str(app_path)).run(timeout=45)
        assert not app.exception
        app.selectbox(key="locality_select").select("Ulwe").run(timeout=30)
        assert not app.exception
