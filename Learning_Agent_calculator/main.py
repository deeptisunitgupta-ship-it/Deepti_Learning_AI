"""
main.py — Entry Point for the LangGraph Calculator Agent

HOW TO RUN:
  python main.py          → runs the demo (no input needed)
  python main.py chat     → starts interactive chat mode
"""

import sys
from langchain_core.messages import HumanMessage
from agent import create_calculator_agent


# ── Display helpers ────────────────────────────────────────────────────────────

def print_banner():
    print("=" * 58)
    print("  LangGraph Calculator Agent — Gemini 2.5 Flash")
    print("=" * 58)


def print_step_trace(messages: list):
    """
    Print every message in the conversation so you can see
    EXACTLY what the LLM reasoned and which tools it called.
    This is the most valuable part for learning!
    """
    print("\n  --- Agent Thought Process ---")
    for msg in messages:
        role = type(msg).__name__
        if role == "HumanMessage":
            print(f"  [YOU]    {msg.content}")
        elif role == "AIMessage":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                    print(f"  [LLM]    Calling tool: {tc['name']}({args})")
            else:
                print(f"  [LLM]    {msg.content}")
        elif role == "ToolMessage":
            print(f"  [TOOL]   Result: {msg.content}")
    print("  " + "-" * 46)


def run_query(agent, question: str):
    """
    Send one question to the agent and print the full trace + final answer.
    """
    print(f"\n  Question: {question}")

    # Invoke the agent with the user's question
    # The agent returns a dict with a "messages" key containing the full conversation
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # Show every step the agent took (tool calls, reasoning, result)
    print_step_trace(result["messages"])

    # The final answer is always the last message from the LLM
    final_answer = result["messages"][-1].content
    print(f"  Answer:   {final_answer}")


# ── Demo mode ──────────────────────────────────────────────────────────────────

def run_demo(agent):
    """Run a set of pre-defined math questions to show all 4 tools."""
    print_banner()
    print("  Running demo questions...\n")

    demo_questions = [
        "What is 125 plus 375?",
        "Subtract 48 from 200.",
        "What is 13 multiplied by 7?",
        "Divide 144 by 12.",
        "I have 3 boxes, each with 24 items. How many items in total?",
    ]

    for question in demo_questions:
        run_query(agent, question)
        print()

    print("=" * 58)
    print("  Demo complete! Run 'python main.py chat' for live chat.")
    print("=" * 58)


# ── Interactive chat mode ──────────────────────────────────────────────────────

def run_chat(agent):
    """Interactive loop — type any math question, type 'quit' to exit."""
    print_banner()
    print("  Type any math question. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("\n  Goodbye!")
            break

        try:
            run_query(agent, user_input)
        except Exception as e:
            print(f"\n  Error: {e}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        agent = create_calculator_agent()
    except ValueError as e:
        print(f"\n  Setup required:\n  {e}\n")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if mode == "chat":
        run_chat(agent)
    else:
        run_demo(agent)
