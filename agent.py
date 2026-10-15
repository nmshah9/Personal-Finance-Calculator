"""
agent.py
========
Agent layer of the Personal Finance AI Agent: LLM + tools + system prompt + a helper
that returns the answer together with a trace of which tools the LLM chose.
"""
from finance_tools import ALL_TOOLS   # (in the notebook, ALL_TOOLS is defined in an earlier cell)

# === SECTION: agent_imports ===
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

load_dotenv()   # reads OPENAI_API_KEY from a local .env file if present


# === SECTION: system_prompt ===
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are FinBot, a Personal Finance Assistant for Indian users. All money is in Indian Rupees (₹).

GOLDEN RULE: never do arithmetic yourself. Every number you report must come from a tool result.

Choose the tool that matches the question:
- emi_calculator       -> loan EMI (needs loan amount, annual interest %, tenure in years)
- sip_calculator       -> future value of a monthly investment (needs monthly amount, annual return %, years)
- goal_sip_calculator  -> monthly amount needed to reach a target corpus (needs target, years, optional return %)
- budget_planner       -> 50-30-20 monthly budget (needs monthly income)
- calculator           -> any other plain arithmetic, such as percentages of an amount

Rules:
1. Convert Indian units before calling a tool: 1 lakh = 100000, 1 crore = 10000000.
2. If a required input is missing, ask ONE short clarifying question instead of guessing.
   Exception: for goal_sip_calculator, if no return rate is given assume 12% p.a. and say clearly that it is an assumption.
3. Answer format: the headline result first, then 2-4 key figures as bullets, then one line stating the formula or assumption used.
4. For investment projections add one short reminder that returns are assumptions, not guarantees.
5. If a tool returns an error, explain it simply and ask for corrected input.
6. If the question is not about personal finance, politely say you can only help with EMI, SIP, savings goals, budgets and calculations.
This is educational guidance, not professional financial advice."""


# === SECTION: build_agent ===
def build_llm(api_key=None, model=DEFAULT_MODEL, temperature=0.0):
    """Create the chat model. temperature=0 keeps tool selection as deterministic as possible."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY (or put it in a .env file).")
    return ChatOpenAI(model=model, temperature=temperature, api_key=key)


def build_agent(api_key=None, llm=None, model=DEFAULT_MODEL):
    """Wire LLM + tools + system prompt into a tool-calling agent (LangChain v1 create_agent)."""
    llm = llm or build_llm(api_key=api_key, model=model)
    return create_agent(model=llm, tools=ALL_TOOLS, system_prompt=SYSTEM_PROMPT)


# === SECTION: ask_helper ===
@dataclass
class ToolTrace:
    """One tool call chosen by the LLM."""
    name: str
    args: dict
    output: str = ""


@dataclass
class AgentResult:
    answer: str
    tool_calls: list = field(default_factory=list)


def _as_text(content: Any) -> str:
    """Message content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        parts.append(block.get("text", "") if isinstance(block, dict) else str(block))
    return "".join(parts)


def ask(agent, question, history=None, recursion_limit=12):
    """Send one question (optionally with earlier chat turns) and return the answer + tool trace."""
    messages = list(history or []) + [HumanMessage(content=question)]
    state = agent.invoke({"messages": messages}, config={"recursion_limit": recursion_limit})
    new_messages = state["messages"][len(messages):]

    traces, by_id = [], {}
    for message in new_messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                trace = ToolTrace(name=call["name"], args=call["args"])
                by_id[call["id"]] = trace
                traces.append(trace)
        elif isinstance(message, ToolMessage) and message.tool_call_id in by_id:
            by_id[message.tool_call_id].output = _as_text(message.content)

    answer = ""
    for message in reversed(new_messages):
        if isinstance(message, AIMessage) and _as_text(message.content).strip():
            answer = _as_text(message.content)
            break
    return AgentResult(answer=answer, tool_calls=traces)
