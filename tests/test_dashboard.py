"""Smoke-test every dashboard page renders without exceptions."""

from pathlib import Path

import pytest
from conftest import requires_model

pytestmark = requires_model
APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = ["Executive overview", "Customer predictor", "Retention planner", "Batch scoring",
         "Model performance", "Explainability"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.header[0].value == page
