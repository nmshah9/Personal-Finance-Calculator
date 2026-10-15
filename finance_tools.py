"""
finance_tools.py
================
Tool layer of the Personal Finance AI Agent.

Design rule: ALL arithmetic lives here, in plain deterministic Python.
The LLM never calculates - it only decides WHICH tool to call and with WHAT inputs.

Layout
  1. Pure functions   -> return dictionaries (easy to unit-test, no LangChain needed)
  2. LangChain tools  -> thin @tool wrappers with strict argument schemas and
                         LLM-facing descriptions (this is what the agent "sees")

The "# === SECTION" markers are used to build the notebook from this very file,
so the notebook and the application can never drift apart.
"""

# === SECTION: imports ===
import ast
import json
import operator
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# === SECTION: helpers ===
MAX_RATE = 100.0      # sanity cap on any annual percentage rate
MAX_YEARS = 100.0     # sanity cap on any duration


def format_indian_number(value, decimals=2):
    """Indian digit grouping: 1234567.5 -> '12,34,567.50'."""
    value = round(float(value), decimals)
    sign = "-" if value < 0 else ""
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return f"{sign}{whole}" + (f".{frac}" if decimals else "")


def format_inr(value, decimals=2):
    """Rupee string with Indian grouping: 1150193.8 -> '₹11,50,193.80'."""
    text = format_indian_number(abs(value), decimals)
    return ("-" if value < 0 else "") + "₹" + text


def _check_positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a number greater than 0 (got {value!r}).")


def _check_rate(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= MAX_RATE:
        raise ValueError(f"{name} must be between 0 and {MAX_RATE:g} percent per year (got {value!r}).")


def _to_months(name, years):
    _check_positive(name, years)
    if years > MAX_YEARS:
        raise ValueError(f"{name} must be at most {MAX_YEARS:g} years (got {years!r}).")
    months = round(years * 12)
    if months < 1:
        raise ValueError(f"{name} must be at least 1 month.")
    return months


# === SECTION: calculator ===
_BINARY_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
MAX_EXPRESSION_LENGTH = 200


def _clean_expression(expression):
    """Normalise things users and LLMs commonly type (₹, commas, ^, '20% of')."""
    text = expression.strip().lower()
    text = text.replace("₹", "").replace(",", "")
    text = text.replace("×", "*").replace("÷", "/").replace("^", "**")
    # "20% of 85000" -> "(20/100)*85000"
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", text)
    return text


def _eval_node(node):
    """Evaluate a parsed expression using an explicit whitelist - never eval()."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression. Use only numbers and + - * / ** % // ( ).")


def safe_calculate(expression):
    """Safely evaluate an arithmetic expression and return a result dictionary."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression is too long (max {MAX_EXPRESSION_LENGTH} characters).")
    cleaned = _clean_expression(expression)
    try:
        result = _eval_node(ast.parse(cleaned, mode="eval"))
    except SyntaxError:
        raise ValueError(f"Could not parse expression: {expression!r}") from None
    except ZeroDivisionError:
        raise ValueError("Division by zero.") from None
    except OverflowError:
        raise ValueError("Result is too large to compute.") from None
    result = round(result, 10)
    is_whole = abs(result - round(result)) < 1e-9
    shown = format_indian_number(result, 0) if is_whole else format_indian_number(result, 4).rstrip("0").rstrip(".")
    return {"expression": expression, "evaluated_as": cleaned,
            "result": int(round(result)) if is_whole else result, "result_formatted": shown}


# === SECTION: emi ===
def calculate_emi(loan_amount, annual_interest_rate, tenure_years):
    """EMI = P * r * (1+r)^n / ((1+r)^n - 1)   with r = annual_rate / 12 / 100,  n = years * 12."""
    _check_positive("loan_amount", loan_amount)
    _check_rate("annual_interest_rate", annual_interest_rate)
    n = _to_months("tenure_years", tenure_years)
    r = annual_interest_rate / 12 / 100
    if r == 0:
        emi = loan_amount / n                                  # interest-free loan
    else:
        growth = (1 + r) ** n
        emi = loan_amount * r * growth / (growth - 1)
    total_payment = emi * n
    total_interest = total_payment - loan_amount
    return {
        "loan_amount": round(loan_amount, 2),
        "annual_interest_rate_percent": annual_interest_rate,
        "tenure_months": n,
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_payment": round(total_payment, 2),
        "summary": (f"EMI of {format_inr(emi)} per month for {n} months on a loan of "
                    f"{format_inr(loan_amount)} at {annual_interest_rate:g}% p.a. "
                    f"Total interest {format_inr(total_interest)}; total repayment {format_inr(total_payment)}."),
    }


# === SECTION: sip ===
def calculate_sip_future_value(monthly_investment, annual_return_rate, years):
    """FV = P * ((1+r)^n - 1) / r   with r = annual_rate / 12 / 100,  n = years * 12."""
    _check_positive("monthly_investment", monthly_investment)
    _check_rate("annual_return_rate", annual_return_rate)
    n = _to_months("years", years)
    r = annual_return_rate / 12 / 100
    future_value = monthly_investment * n if r == 0 else monthly_investment * ((1 + r) ** n - 1) / r
    invested = monthly_investment * n
    gains = future_value - invested
    return {
        "monthly_investment": round(monthly_investment, 2),
        "annual_return_rate_percent": annual_return_rate,
        "tenure_months": n,
        "total_invested": round(invested, 2),
        "estimated_gains": round(gains, 2),
        "future_value": round(future_value, 2),
        "summary": (f"Investing {format_inr(monthly_investment)} per month for {n} months at an assumed "
                    f"{annual_return_rate:g}% p.a. grows to about {format_inr(future_value)}: "
                    f"{format_inr(invested)} invested + {format_inr(gains)} estimated gains."),
    }


# === SECTION: goal_sip ===
def calculate_required_sip(target_amount, years, annual_return_rate=12.0):
    """Inverse of the SIP formula:  P = FV * r / ((1+r)^n - 1)."""
    _check_positive("target_amount", target_amount)
    _check_rate("annual_return_rate", annual_return_rate)
    n = _to_months("years", years)
    r = annual_return_rate / 12 / 100
    monthly = target_amount / n if r == 0 else target_amount * r / ((1 + r) ** n - 1)
    invested = monthly * n
    return {
        "target_amount": round(target_amount, 2),
        "annual_return_rate_percent": annual_return_rate,
        "tenure_months": n,
        "required_monthly_investment": round(monthly, 2),
        "total_invested": round(invested, 2),
        "estimated_gains": round(target_amount - invested, 2),
        "summary": (f"To reach {format_inr(target_amount)} in {n} months at an assumed {annual_return_rate:g}% p.a., "
                    f"invest about {format_inr(monthly)} per month "
                    f"({format_inr(invested)} contributed + {format_inr(target_amount - invested)} estimated gains)."),
    }


# === SECTION: budget ===
BUDGET_RULE = {"Needs": 50, "Wants": 30, "Savings": 20}          # the 50-30-20 rule (% of income)
BUDGET_BREAKDOWN = {                                            # illustrative split INSIDE each bucket
    "Needs": {"Rent / home-loan EMI": 50, "Groceries": 20, "Utilities & bills": 10,
              "Transport": 10, "Insurance & health": 10},
    "Wants": {"Dining & entertainment": 40, "Shopping & lifestyle": 35, "Travel & hobbies": 25},
    "Savings": {"SIP / mutual funds": 50, "Emergency fund": 30, "Retirement (PPF / NPS)": 20},
}


def plan_budget(monthly_income):
    """50-30-20 budget: 50% Needs, 30% Wants, 20% Savings, plus an illustrative sub-split."""
    _check_positive("monthly_income", monthly_income)
    buckets = {}
    for bucket, percent in BUDGET_RULE.items():
        amount = round(monthly_income * percent / 100, 2)
        items, running = {}, 0.0
        names = list(BUDGET_BREAKDOWN[bucket])
        for name in names[:-1]:
            value = round(amount * BUDGET_BREAKDOWN[bucket][name] / 100, 2)
            items[name] = value
            running += value
        items[names[-1]] = round(amount - running, 2)           # last line absorbs rounding
        buckets[bucket] = {"percent": percent, "amount": amount, "suggested_breakdown": items}
    return {
        "monthly_income": round(monthly_income, 2),
        "rule": "50-30-20",
        "buckets": buckets,
        "summary": (f"For a monthly income of {format_inr(monthly_income)}: "
                    + ", ".join(f"{b} {format_inr(v['amount'])} ({v['percent']}%)" for b, v in buckets.items()) + "."),
    }


# === SECTION: langchain_tools ===
def _run(function, **kwargs):
    """Run a pure function and return JSON text; turn bad input into a readable error the agent can act on."""
    try:
        return json.dumps(function(**kwargs), ensure_ascii=False)
    except ValueError as error:
        return json.dumps({"error": str(error)}, ensure_ascii=False)


class CalculatorInput(BaseModel):
    expression: str = Field(description=(
        "Arithmetic expression using numbers and + - * / ** % // ( ). "
        "Convert percentages to decimals or use 'X% of Y'. Example: '0.20 * 85000'."))


class EMIInput(BaseModel):
    loan_amount: float = Field(description="Principal loan amount in rupees (convert lakh/crore first: 10 lakh = 1000000).")
    annual_interest_rate: float = Field(description="Annual interest rate in percent, e.g. 8.5 for 8.5% p.a.")
    tenure_years: float = Field(description="Loan tenure in years (convert months to years, e.g. 18 months = 1.5).")


class SIPInput(BaseModel):
    monthly_investment: float = Field(description="Amount invested every month, in rupees.")
    annual_return_rate: float = Field(description="Expected annual return in percent, e.g. 12 for 12% p.a.")
    years: float = Field(description="Investment duration in years.")


class GoalSIPInput(BaseModel):
    target_amount: float = Field(description="Target corpus in rupees (convert lakh/crore first: 20 lakh = 2000000).")
    years: float = Field(description="Number of years available to reach the target.")
    annual_return_rate: float = Field(default=12.0, description=(
        "Expected annual return in percent. If the user did not state one, use 12 and tell the user it is an assumption."))


class BudgetInput(BaseModel):
    monthly_income: float = Field(description="Monthly take-home income in rupees.")


@tool("calculator", args_schema=CalculatorInput)
def calculator_tool(expression: str) -> str:
    """General-purpose calculator for plain arithmetic: percentages, sums, differences, products,
    ratios, powers. Use it ONLY for generic maths. Do NOT use it for EMI, SIP / investment growth,
    savings goals or budgets - those have dedicated tools."""
    return _run(safe_calculate, expression=expression)


@tool("emi_calculator", args_schema=EMIInput)
def emi_tool(loan_amount: float, annual_interest_rate: float, tenure_years: float) -> str:
    """Calculate the monthly EMI (equated monthly instalment) of a loan, plus total interest and total
    repayment. Use for home / car / personal loan questions given loan amount, interest rate and tenure."""
    return _run(calculate_emi, loan_amount=loan_amount,
                annual_interest_rate=annual_interest_rate, tenure_years=tenure_years)


@tool("sip_calculator", args_schema=SIPInput)
def sip_tool(monthly_investment: float, annual_return_rate: float, years: float) -> str:
    """Calculate the FUTURE VALUE of a monthly investment (SIP) given the monthly amount, expected annual
    return and duration. Use for 'if I invest X per month for N years at R%, how much will I have?'."""
    return _run(calculate_sip_future_value, monthly_investment=monthly_investment,
                annual_return_rate=annual_return_rate, years=years)


@tool("goal_sip_calculator", args_schema=GoalSIPInput)
def goal_sip_tool(target_amount: float, years: float, annual_return_rate: float = 12.0) -> str:
    """Calculate the MONTHLY amount that must be invested to reach a target corpus in a given number of
    years. Use for 'how much should I save monthly to reach X in N years?'."""
    return _run(calculate_required_sip, target_amount=target_amount, years=years,
                annual_return_rate=annual_return_rate)


@tool("budget_planner", args_schema=BudgetInput)
def budget_tool(monthly_income: float) -> str:
    """Create a monthly budget from income using the 50-30-20 rule (50% needs, 30% wants, 20% savings)
    with a suggested split inside each bucket. Use for 'create a budget / plan my spending' requests."""
    return _run(plan_budget, monthly_income=monthly_income)


ALL_TOOLS = [calculator_tool, emi_tool, sip_tool, goal_sip_tool, budget_tool]
