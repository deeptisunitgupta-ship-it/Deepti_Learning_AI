"""
agent.py — LangGraph Calculator Agent with Google Gemini 2.5 Flash

WHAT IS THIS FILE?
------------------
This file creates the LangGraph agent — the "brain" of the application.

HOW A LANGGRAPH REACT AGENT WORKS:
------------------------------------
ReAct = "Reasoning + Acting"

The agent follows this loop:
  1. REASON  — LLM reads the user's question and thinks: "What tool should I use?"
  2. ACT     — LLM calls a tool (e.g., multiply(a=6, b=7))
  3. OBSERVE — The tool runs and returns the result (e.g., 42)
  4. REASON  — LLM reads the result and decides: "Do I need more steps, or am I done?"
  5. RESPOND — LLM generates a final natural language answer

LANGGRAPH UNDER THE HOOD:
--------------------------
create_react_agent() builds this graph automatically:

  START
    |
    v
  [agent node]  ← LLM decides: call a tool or give final answer
    |       |
    |   (tool call)
    |       v
    |  [tools node]  ← Python tool function runs here
    |       |
    |<------+  (tool result fed back to LLM)
    |
  (final answer)
    |
    v
   END

The STATE is a list of messages that grows with each step:
  HumanMessage("What is 6 * 7?")
  AIMessage(tool_call: multiply(6, 7))
  ToolMessage(result: 42.0)
  AIMessage("6 multiplied by 7 is 42.")
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from tools import CALCULATOR_TOOLS

# Load the GOOGLE_API_KEY from the .env file
load_dotenv()


def create_calculator_agent():
    """
    Create and return the LangGraph calculator agent.

    Returns:
        A compiled LangGraph graph (the agent) ready to be invoked.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Paste your API key from https://aistudio.google.com/app/apikey\n"
            "  3. Run again."
        )

    # ── Create the Gemini 2.5 Flash LLM ──────────────────────────────────────
    # model="gemini-2.5-flash"  →  Google's fast, capable model
    # temperature=0             →  Deterministic responses (good for math)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    # ── Create the ReAct Agent ────────────────────────────────────────────────
    # create_react_agent() takes the LLM and the list of tools.
    # It automatically builds the full LangGraph with:
    #   - an agent node (LLM decides what to do)
    #   - a tools node (executes the chosen tool)
    #   - conditional edges (loop back after tool use, or end)
    agent = create_react_agent(
        model=llm,
        tools=CALCULATOR_TOOLS,
        prompt=(
            "You are a helpful calculator assistant. "
            "Use the available tools to perform arithmetic calculations. "
            "Always use a tool to compute the answer — do not calculate mentally. "
            "After getting the tool result, explain the answer clearly."
        ),
    )

    return agent
