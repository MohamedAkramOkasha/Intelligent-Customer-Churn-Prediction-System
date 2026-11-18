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


def test_sidebar_cost_moves_profit_threshold():
    """The profit threshold must react to the shared campaign assumptions in the sidebar."""
    from streamlit.testing.v1 import AppTest

    def profit_card(at):
        return next(m.value for m in at.markdown if m.value.startswith("**Profit-optimal threshold"))

    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Model performance").run()
    cheap = profit_card(at)
    at.sidebar.number_input[0].set_value(250.0).run()  # very expensive contacts
    assert not at.exception
    pricey = profit_card(at)
    t_cheap, t_pricey = (float(s.split(": ")[1].strip("*")) for s in (cheap, pricey))
    assert t_pricey > t_cheap
