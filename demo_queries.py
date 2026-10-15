"""demo_queries.py - the test queries used by the demo runner, the notebook, the Streamlit app and the tests."""

# === SECTION: demo_queries ===
# (query, tool the agent is expected to choose)
REQUIRED_QUERIES = [
    ("Calculate EMI for ₹10 lakh loan at 8% interest for 20 years", "emi_calculator"),
    ("If I invest ₹7000 monthly for 15 years at 10% return", "sip_calculator"),
    ("Create a budget plan for income ₹90,000", "budget_planner"),
    ("What is 20% of ₹85,000", "calculator"),
    ("How much will I get if I invest ₹3000 per month for 5 years at 8% return", "sip_calculator"),
]

EXTRA_QUERIES = [
    ("If I invest ₹5000 per month for 10 years at 12% return, how much money will I have?", "sip_calculator"),
    ("Calculate EMI for a ₹10 lakh loan at 9% interest for 20 years.", "emi_calculator"),
    ("How much should I save monthly to reach ₹20 lakh in 10 years?", "goal_sip_calculator"),
    ("Create a monthly budget for someone earning ₹80,000.", "budget_planner"),
]
