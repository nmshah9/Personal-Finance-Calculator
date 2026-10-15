"""app.py - Streamlit front-end for the Personal Finance AI Agent.   Run:  streamlit run app.py

Two modes:
  * Calculators   - visitors type amounts and instantly see the outcome (no API key needed)
  * AI Agent Chat - ask in plain English; the LLM picks the tool (needs an OpenAI key)
"""
import os

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import DEFAULT_MODEL, ask, build_agent
from demo_queries import REQUIRED_QUERIES
from finance_tools import (ALL_TOOLS, calculate_emi, calculate_required_sip,
                           calculate_sip_future_value, format_inr, plan_budget, safe_calculate)

# ============================================================
# LOAD BANNER
# ============================================================
from PIL import Image
banner = Image.open("banner.png")
st.image(banner, use_column_width=True)

# ================================================================

st.set_page_config(page_title="Personal Finance AI Agent", page_icon="💰", layout="wide")

MODE_CALC = "🧮 Calculators"
MODE_CHAT = "💬 AI Agent Chat"


# =============================================================== helpers
def resolve_api_key() -> str:
    """Priority: key typed in the sidebar > Streamlit secrets > environment variable."""
    typed = st.session_state.get("typed_key", "").strip()
    if typed:
        return typed
    try:
        secret = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:          # no secrets.toml present
        secret = ""
    return secret or os.getenv("OPENAI_API_KEY", "")


@st.cache_resource(show_spinner=False)
def get_agent(api_key: str):
    return build_agent(api_key=api_key)


def money_input(label, value, step, key):
    amount = st.number_input(label, min_value=0, value=int(value), step=int(step), key=key)
    st.caption(f"= {format_inr(amount, 0)}")
    return amount


def rate_input(label, value, key):
    return st.number_input(label, min_value=0.0, max_value=100.0, value=float(value), step=0.1, key=key)


def loan_balance_by_year(loan, annual_rate, emi, months):
    r = annual_rate / 12 / 100
    rows = []
    for k in list(range(0, months, 12)) + [months]:
        balance = loan - emi * k if r == 0 else loan * (1 + r) ** k - emi * ((1 + r) ** k - 1) / r
        rows.append({"Year": round(k / 12, 2), "Outstanding balance": max(balance, 0.0)})
    return pd.DataFrame(rows).set_index("Year")


def sip_growth_by_year(monthly, annual_rate, years):
    rows = [{"Year": 0, "Total invested": 0.0, "Corpus value": 0.0}]
    for year in range(1, years + 1):
        point = calculate_sip_future_value(monthly, annual_rate, year)
        rows.append({"Year": year, "Total invested": point["total_invested"], "Corpus value": point["future_value"]})
    return pd.DataFrame(rows).set_index("Year")


# =============================================================== calculators
def render_calculators():
    st.write("Enter your own numbers and see the result instantly. Every figure is computed by the same "
             "verified tools the AI agent uses.")
    emi_tab, sip_tab, goal_tab, budget_tab, calc_tab = st.tabs(
        ["🏠 EMI", "📈 SIP returns", "🎯 Savings goal", "🧾 Budget (50-30-20)", "🔢 Calculator"])

    with emi_tab:
        left, right = st.columns([1, 2])
        with left:
            loan = money_input("Loan amount (₹)", 1_000_000, 50_000, "emi_loan")
            rate = rate_input("Interest rate (% per year)", 8.5, "emi_rate")
            years = st.number_input("Tenure (years)", min_value=1, max_value=40, value=20, step=1, key="emi_years")
        with right:
            try:
                result = calculate_emi(loan, rate, years)
                a, b, c = st.columns(3)
                a.metric("Monthly EMI", format_inr(result["monthly_emi"]))
                b.metric("Total interest", format_inr(result["total_interest"]))
                c.metric("Total repayment", format_inr(result["total_payment"]))
                st.caption(result["summary"])
                st.subheader("Outstanding loan balance")
                st.line_chart(loan_balance_by_year(loan, rate, result["monthly_emi"], result["tenure_months"]))
            except ValueError as error:
                st.info(f"Enter a loan amount above zero. ({error})")

    with sip_tab:
        left, right = st.columns([1, 2])
        with left:
            monthly = money_input("Monthly investment (₹)", 5_000, 500, "sip_monthly")
            rate = rate_input("Expected return (% per year)", 12, "sip_rate")
            years = st.number_input("Duration (years)", min_value=1, max_value=50, value=10, step=1, key="sip_years")
        with right:
            try:
                result = calculate_sip_future_value(monthly, rate, years)
                a, b, c = st.columns(3)
                a.metric("Future value", format_inr(result["future_value"]))
                b.metric("Total invested", format_inr(result["total_invested"]))
                c.metric("Estimated gains", format_inr(result["estimated_gains"]))
                st.caption(result["summary"] + " Returns are assumptions, not guarantees.")
                st.subheader("Growth of your investment")
                st.line_chart(sip_growth_by_year(monthly, rate, years))
            except ValueError as error:
                st.info(f"Enter a monthly amount above zero. ({error})")

    with goal_tab:
        left, right = st.columns([1, 2])
        with left:
            target = money_input("Target amount (₹)", 2_000_000, 100_000, "goal_target")
            years = st.number_input("Years to reach it", min_value=1, max_value=50, value=10, step=1, key="goal_years")
            rate = rate_input("Expected return (% per year)", 12, "goal_rate")
        with right:
            try:
                result = calculate_required_sip(target, years, rate)
                a, b, c = st.columns(3)
                a.metric("Save every month", format_inr(result["required_monthly_investment"]))
                b.metric("Total you invest", format_inr(result["total_invested"]))
                c.metric("Estimated gains", format_inr(result["estimated_gains"]))
                st.caption(result["summary"])
                st.subheader("Path to your goal")
                st.line_chart(sip_growth_by_year(result["required_monthly_investment"], rate, years))
            except ValueError as error:
                st.info(f"Enter a target amount above zero. ({error})")

    with budget_tab:
        income = money_input("Monthly income (₹)", 80_000, 5_000, "budget_income")
        try:
            plan = plan_budget(income)
            columns = st.columns(3)
            for column, (bucket, info) in zip(columns, plan["buckets"].items()):
                with column:
                    st.metric(f"{bucket} ({info['percent']}%)", format_inr(info["amount"]))
                    table = pd.DataFrame({"Item": list(info["suggested_breakdown"]),
                                          "Amount": [format_inr(v) for v in info["suggested_breakdown"].values()]})
                    st.dataframe(table, hide_index=True, width="stretch")
            st.caption("50-30-20 rule: needs / wants / savings. The split inside each bucket is illustrative.")
        except ValueError as error:
            st.info(f"Enter an income above zero. ({error})")

    with calc_tab:
        expression = st.text_input("Expression", value="20% of 85000", key="calc_expr",
                                   help="Examples: 20% of 85000   |   (1+0.08)**5   |   2500*12")
        try:
            result = safe_calculate(expression)
            st.metric("Result", result["result_formatted"])
            st.caption(f"Evaluated as: {result['evaluated_as']}")
        except ValueError as error:
            st.warning(str(error))


# =============================================================== chat
def render_tools(tools):
    if tools:
        with st.expander(f"🔧 Tools used ({len(tools)})"):
            for call in tools:
                st.markdown(f"**{call['name']}**")
                st.json(call["args"])
                st.code(call["output"], language="json")


def render_chat():
    st.write("Ask in plain English. The LLM decides which tool to use; the tools do the maths.")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_tools(message.get("tools"))

    prompt = st.chat_input("e.g. Calculate EMI for a ₹25 lakh loan at 8.5% for 15 years")
    if st.session_state.get("pending_query"):
        prompt = st.session_state.pop("pending_query")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        api_key = resolve_api_key()
        if not api_key:
            reply, tools = "Please enter your OpenAI API key in the sidebar to use the AI chat.", []
            st.warning(reply)
        else:
            history = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state["messages"][:-1]
            ]
            try:
                with st.spinner("Thinking..."):
                    result = ask(get_agent(api_key), prompt, history=history)
                reply = result.answer
                tools = [{"name": c.name, "args": c.args, "output": c.output} for c in result.tool_calls]
                st.markdown(reply)
                render_tools(tools)
            except Exception as error:   # bad key, network, rate limit...
                reply, tools = f"Sorry, something went wrong: {error}", []
                st.error(reply)
    st.session_state["messages"].append({"role": "assistant", "content": reply, "tools": tools})


# =============================================================== sidebar
with st.sidebar:
    st.header("AI chat settings")
    st.text_input("OpenAI API key", type="password", key="typed_key",
                  help="Only needed for the AI chat. Calculators work without it. "
                       "On Streamlit Cloud you can store the key in Secrets instead.")
    st.caption(f"Model: `{DEFAULT_MODEL}`")

    st.header("Try an AI example")
    for index, (query, _) in enumerate(REQUIRED_QUERIES):
        if st.button(query, key=f"example_{index}", width="stretch"):
            st.session_state["pending_query"] = query
            st.session_state["mode"] = MODE_CHAT          # jump to the chat view

    if st.button("Clear conversation", width="stretch"):
        st.session_state["messages"] = []
        st.rerun()

    with st.expander("Tools available to the agent"):
        for t in ALL_TOOLS:
            st.markdown(f"**`{t.name}`** - {t.description.split('.')[0]}.")
    st.caption("Educational tool - not professional financial advice. Returns are assumptions, not guarantees.")

# =============================================================== main
st.title("💰 Personal Finance AI Agent")
mode = st.radio("Mode", [MODE_CALC, MODE_CHAT], horizontal=True, key="mode", label_visibility="collapsed")
if mode == MODE_CALC:
    render_calculators()
else:
    render_chat()
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style='text-align: center;'>

### 🧮 Personal-Finance-Calculator-using-Agentic AI ###

Built with ❤️ using Streamlit | Developed by nmshah9

</div>
""", unsafe_allow_html=True)
