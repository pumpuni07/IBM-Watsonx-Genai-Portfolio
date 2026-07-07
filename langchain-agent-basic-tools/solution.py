"""
Exercise 7 — Creating Your First LangChain Agent with Basic Tools
Course: Build Smarter AI Apps: Empower LLMs with LangChain (IBM Skills Network)

Solution by Jack Pumpuni Frimpong-Manso (github.com/pumpuni07)

A ReAct agent with two custom tools:
  1. Calculator  — safe arithmetic evaluator (AST-based, no eval())
  2. TextFormatter — uppercase / lowercase / titlecase conversion

Requires the classic LangChain agent stack (see requirements.txt):
  langchain==0.3.27, langchain-ibm==0.3.15
The lab's create_react_agent / AgentExecutor API was removed in LangChain 1.x.
"""

import ast
import operator
import os

from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# Tool 1: Calculator
# ---------------------------------------------------------------------------
# The lab hint suggests eval(), but eval() executes arbitrary code — a real
# security risk when input comes from an LLM. This solution parses the
# expression into an AST and only permits arithmetic nodes.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST limited to basic arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """A simple calculator that can add, subtract, multiply, or divide two numbers.
    Input should be a mathematical expression like '2 + 2' or '15 / 3'."""
    try:
        tree = ast.parse(expression.strip().strip("'\""), mode="eval")
        result = _safe_eval(tree)
        # Show integers without a trailing .0
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"The result of {expression.strip()} is {result}"
    except ZeroDivisionError:
        return "Error calculating: division by zero"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 2: Text formatter
# ---------------------------------------------------------------------------
def format_text(text: str) -> str:
    """Format text to uppercase, lowercase, or title case.
    Input should be in format: '[format_type]: [text]'
    where format_type is 'uppercase', 'lowercase', or 'titlecase'."""
    try:
        if ":" not in text:
            return "Error formatting text: input must be '[format_type]: [text]'"
        format_type, _, content = text.partition(":")
        format_type = format_type.strip().strip("'\"").lower()
        content = content.strip().strip("'\"")

        formatters = {
            "uppercase": str.upper,
            "lowercase": str.lower,
            "titlecase": str.title,
        }
        if format_type not in formatters:
            return (
                f"Error formatting text: unknown format '{format_type}'. "
                "Use 'uppercase', 'lowercase', or 'titlecase'."
            )
        return formatters[format_type](content)
    except Exception as e:
        return f"Error formatting text: {str(e)}"


# ---------------------------------------------------------------------------
# Wrap functions as LangChain Tool objects
# ---------------------------------------------------------------------------
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description=(
            "Performs basic arithmetic (+, -, *, /, %, **). "
            "Input: a plain mathematical expression such as '25 + 63' or '15 / 3'."
        ),
    ),
    Tool(
        name="TextFormatter",
        func=format_text,
        description=(
            "Converts text case. Input format: '[format_type]: [text]' where "
            "format_type is 'uppercase', 'lowercase', or 'titlecase'. "
            "Example: 'uppercase: hello world'."
        ),
    ),
]


# ---------------------------------------------------------------------------
# ReAct prompt template
# ---------------------------------------------------------------------------
# create_react_agent requires the variables: {tools}, {tool_names}, {input},
# and {agent_scratchpad}.
prompt_template = """You are a helpful assistant who can use tools to help with simple tasks.
You have access to these tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(prompt_template)


# ---------------------------------------------------------------------------
# LLM, agent, and executor
# ---------------------------------------------------------------------------
def build_llm():
    """Initialize the watsonx.ai LLM.

    Inside the IBM Skills Network lab environment this works without an API
    key (project_id='skills-network'). Elsewhere, export WATSONX_APIKEY and
    WATSONX_PROJECT_ID from your IBM Cloud account. If the model ID below is
    deprecated, substitute a current instruct model from the watsonx catalog.
    """
    from langchain_ibm import WatsonxLLM

    return WatsonxLLM(
        model_id="ibm/granite-3-3-8b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=os.environ.get("WATSONX_PROJECT_ID", "skills-network"),
        params={
            "decoding_method": "greedy",
            "max_new_tokens": 256,
            "stop_sequences": ["\nObservation"],
        },
    )


def build_agent_executor(llm) -> AgentExecutor:
    """Assemble the ReAct agent and its executor."""
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test_questions = [
    "What is 25 + 63?",
    "Can you convert 'hello world' to uppercase?",
    "Calculate 15 * 7",
    "titlecase: langchain is awesome",
]


def run_tests(agent_executor: AgentExecutor) -> None:
    for question in test_questions:
        print(f"\n===== Testing: {question} =====")
        result = agent_executor.invoke({"input": question})
        print(f"Answer: {result['output']}")


if __name__ == "__main__":
    llm = build_llm()
    agent_executor = build_agent_executor(llm)
    run_tests(agent_executor)
