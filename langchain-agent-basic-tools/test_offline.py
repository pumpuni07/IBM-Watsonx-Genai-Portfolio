"""
Offline verification of solution.py — no watsonx credentials required.

A FakeListLLM returns scripted ReAct-format responses, so the complete
agent loop (prompt -> agent -> tool call -> observation -> final answer)
is exercised end-to-end. Also unit-tests both tools directly.

Run:  python test_offline.py
"""

import sys
from solution import (
    calculator,
    format_text,
    tools,
    prompt,
    build_agent_executor,
    test_questions,
)

failures = []


def check(label, actual, expected_substring):
    ok = expected_substring in actual
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {actual!r}")
    if not ok:
        failures.append(label)


# ---------------------------------------------------------------- tool units
print("--- Tool unit tests ---")
check("calc add", calculator("25 + 63"), "88")
check("calc mult", calculator("15 * 7"), "105")
check("calc div", calculator("15 / 3"), "5")
check("calc zero div", calculator("1 / 0"), "division by zero")
check("calc injection blocked", calculator("__import__('os')"), "Error calculating")
check("fmt upper", format_text("uppercase: hello world"), "HELLO WORLD")
check("fmt lower", format_text("lowercase: HELLO"), "hello")
check("fmt title", format_text("titlecase: langchain is awesome"), "Langchain Is Awesome")
check("fmt bad type", format_text("reverse: abc"), "unknown format")
check("fmt no colon", format_text("just text"), "Error formatting")

# ---------------------------------------------------------------- agent loop
print("\n--- Full agent loop with scripted fake LLM ---")
from langchain_core.language_models import FakeListLLM

# Scripted ReAct responses: for each question, one tool call then a final answer.
responses = [
    "I need to add these numbers.\nAction: Calculator\nAction Input: 25 + 63",
    "I now know the final answer.\nFinal Answer: 25 + 63 = 88",
    "I should format this text.\nAction: TextFormatter\nAction Input: uppercase: hello world",
    "I now know the final answer.\nFinal Answer: HELLO WORLD",
    "I need to multiply.\nAction: Calculator\nAction Input: 15 * 7",
    "I now know the final answer.\nFinal Answer: 15 * 7 = 105",
    "I should format this text.\nAction: TextFormatter\nAction Input: titlecase: langchain is awesome",
    "I now know the final answer.\nFinal Answer: Langchain Is Awesome",
]

fake_llm = FakeListLLM(responses=responses)
agent_executor = build_agent_executor(fake_llm)

expected = ["88", "HELLO WORLD", "105", "Langchain Is Awesome"]
for question, exp in zip(test_questions, expected):
    print(f"\n===== Testing: {question} =====")
    result = agent_executor.invoke({"input": question})
    check(f"agent: {question}", result["output"], exp)

# ---------------------------------------------------------------- summary
print("\n--- Summary ---")
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("All offline tests passed: tools, prompt, agent construction, and full ReAct loop.")
