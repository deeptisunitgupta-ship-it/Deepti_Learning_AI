"""
tools.py — Calculator Tools for the LangGraph Agent

WHAT IS THIS FILE?
------------------
This file defines the 4 calculator tools that the AI agent can use.

WHAT IS A TOOL (in LangChain / LangGraph)?
-------------------------------------------
A Tool is a normal Python function that the LLM can choose to call.

When you ask the agent: "What is 25 multiplied by 4?"
The LLM reads the tool descriptions and decides:
  → "I should call the `multiply` tool with a=25, b=4"
The agent executes the function and gives the result back to the LLM.
The LLM then responds: "25 multiplied by 4 is 100."

KEY POINTS:
  - @tool decorator makes any Python function a callable LLM tool
  - The function's DOCSTRING is what the LLM reads to understand the tool
  - Type hints (float) tell LangChain what arguments to expect
  - The LLM never sees the code — only the name, docstring, and argument types
"""

from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """
    Add two numbers together and return the result.
    Use this tool when the user wants to add, sum, or find the total of two numbers.
    Example: 'What is 5 plus 3?' or 'Add 10 and 20'
    """
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """
    Subtract the second number (b) from the first number (a) and return the result.
    Use this tool when the user wants to subtract, find the difference, or take away.
    Example: 'What is 10 minus 4?' or 'Subtract 3 from 15'
    """
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together and return the result.
    Use this tool when the user wants to multiply, find the product, or scale a number.
    Example: 'What is 6 times 7?' or 'Multiply 5 by 8'
    """
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """
    Divide the first number (a) by the second number (b) and return the result.
    Use this tool when the user wants to divide, find the quotient, or split a number.
    Example: 'What is 20 divided by 4?' or 'Divide 100 by 5'
    IMPORTANT: b (the divisor) must not be zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero. Please provide a non-zero divisor.")
    return a / b


# List of all tools — passed to the agent
CALCULATOR_TOOLS = [add, subtract, multiply, divide]
