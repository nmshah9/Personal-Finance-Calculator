"""Headless Streamlit test of app.py with the scripted model patched in (no network)."""
from pathlib import Path

import agent as agent_module
from streamlit.testing.v1 import AppTest

from tests.fake_llm import ScriptedToolModel


MODE_CALC, MODE_CHAT = "🧮 Calculators", "💬 AI Agent Chat"
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def make_app(monkeypatch):
    monkeypatch.setattr(agent_module, "build_llm", lambda *a, **k: ScriptedToolModel())
    app = AppTest.from_file(str(APP_PATH), default_timeout=30)
    app.run()
    return app


def test_app_loads_without_error(monkeypatch):
    app = make_app(monkeypatch)
    assert not app.exception
    assert app.title[0].value.endswith("Personal Finance AI Agent")


def test_missing_key_shows_warning(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = make_app(monkeypatch)
    app.radio(key="mode").set_value(MODE_CHAT).run()
    app.chat_input[0].set_value("What is 2+2").run()
    assert any("API key" in w.value for w in app.warning)


def test_chat_flow_with_tool_trace(monkeypatch):
    app = make_app(monkeypatch)
    app.radio(key="mode").set_value(MODE_CHAT).run()
    app.sidebar.text_input(key="typed_key").set_value("sk-test").run()
    app.chat_input[0].set_value("Calculate EMI for ₹10 lakh loan at 8% interest for 20 years").run()
    assert not app.exception
    assistant_text = " ".join(m.value for m in app.markdown)
    assert "8,364.40" in assistant_text
    assert any("Tools used" in e.label for e in app.expander)


def test_example_button_runs_query(monkeypatch):
    app = make_app(monkeypatch)
    app.sidebar.text_input(key="typed_key").set_value("sk-test").run()
    app.sidebar.button(key="example_3").click().run()           # "What is 20% of ₹85,000"
    assert not app.exception
    assert "17,000" in " ".join(m.value for m in app.markdown)


def metrics(app):
    return {m.label: m.value for m in app.metric}


def test_calculators_are_default_and_need_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = make_app(monkeypatch)
    assert app.radio(key="mode").value == MODE_CALC
    assert not app.exception and not app.chat_input
    assert metrics(app)["Needs (50%)"] == "₹40,000.00"          # default budget for ₹80,000


def test_emi_calculator_user_input(monkeypatch):
    app = make_app(monkeypatch)
    app.number_input(key="emi_rate").set_value(8.0).run()        # default loan ₹10 lakh, 20 years
    assert metrics(app)["Monthly EMI"] == "₹8,364.40"
    app.number_input(key="emi_loan").set_value(0).run()          # invalid amount -> friendly message, no crash
    assert not app.exception and any("loan amount" in i.value for i in app.info)


def test_sip_and_goal_calculators_user_input(monkeypatch):
    app = make_app(monkeypatch)
    assert metrics(app)["Future value"] == "₹11,50,193.45"       # defaults: ₹5,000, 12%, 10 years
    assert metrics(app)["Save every month"] == "₹8,694.19"       # defaults: ₹20 lakh, 10 years, 12%
    app.number_input(key="sip_monthly").set_value(7000).run()
    app.number_input(key="sip_rate").set_value(10.0).run()
    app.number_input(key="sip_years").set_value(15).run()
    assert metrics(app)["Future value"] == "₹29,01,292.42"


def test_budget_and_expression_calculator(monkeypatch):
    app = make_app(monkeypatch)
    app.number_input(key="budget_income").set_value(90000).run()
    assert metrics(app)["Savings (20%)"] == "₹18,000.00"
    app.text_input(key="calc_expr").set_value("20% of 85000").run()
    assert metrics(app)["Result"] == "17,000"
    app.text_input(key="calc_expr").set_value("__import__('os')").run()
    assert not app.exception and app.warning
