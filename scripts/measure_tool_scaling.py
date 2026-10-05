"""Measures how FunctionGemma 270M tool-call accuracy varies with the number of
tools offered, and records the schema the mobile-actions finetune actually
follows.

Run: venv/bin/python scripts/measure_tool_scaling.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import litert_lm
from litert_lm import interfaces

MODEL = "assets/function-gemma-q8-ekv1024.litertlm"

# The eight actions in the agent's ActionRegistry, described in the wording the
# mobile-actions finetune responds to. Names and descriptions are the variable
# under test, not the agent's original wording.
TOOL_SPECS: dict[str, tuple[str, dict[str, str]]] = {
    "open_flashlight": ("Turns the phone's flashlight on.", {}),
    "close_flashlight": ("Turns the phone's flashlight off.", {}),
    "query_calendar": ("Lists the events on the user's calendar for today.", {}),
    "take_photo": ("Takes a photo with the phone's camera.", {}),
    "set_alarm": (
        "Sets an alarm on the phone.",
        {"time": "The 24-hour alarm time as HH:MM."},
    ),
    "send_message": (
        "Sends a text message to a contact.",
        {
            "contact_name": "Name of the contact to message.",
            "message": "The text of the message to send.",
        },
    ),
    "append_note": (
        "Appends a note with a title to the notes file.",
        {"title": "The title of the note."},
    ),
    "noop": ("Does nothing.", {}),
}

# Prompt -> (expected tool, expected argument subset).
# The expectation is intent, not a fixed phrasing: each prompt has a paraphrased
# twin so a template-matching model cannot score well by accident.
CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("turn on the flashlight", "open_flashlight", {}),
    ("switch the flashlight on please", "open_flashlight", {}),
    ("i need some light in here", "open_flashlight", {}),
    ("turn off the flashlight", "close_flashlight", {}),
    ("kill the torch now", "close_flashlight", {}),
    ("what is on my calendar today", "query_calendar", {}),
    ("do i have any meetings today", "query_calendar", {}),
    ("take a photo", "take_photo", {}),
    ("snap a picture of the room", "take_photo", {}),
    ("set an alarm for 07:30", "set_alarm", {"time": "07:30"}),
    ("wake me up at 06:00", "set_alarm", {"time": "06:00"}),
    ("send a message to mom saying hi", "send_message", {"contact_name": "mom"}),
    ("text alex that i am running late", "send_message", {"contact_name": "alex"}),
    ("write a note titled groceries", "append_note", {"title": "groceries"}),
    ("remind me to buy milk", "append_note", {"title": "milk"}),
    ("what is the weather in nairobi", "noop", {}),
    ("thanks, that was helpful", "noop", {}),
]

# Tool sets to test. Order is the priority order the agent would use; each
# prefix keeps the flash/calendar pair first because those are the only ones
# that resolve reliably, which the prefixes are meant to expose.
TOOL_SETS: dict[str, list[str]] = {
    "1_tool": ["query_calendar"],
    "2_tools": ["open_flashlight", "query_calendar"],
    "4_tools": ["open_flashlight", "close_flashlight", "query_calendar", "take_photo"],
    "8_tools": list(TOOL_SPECS),
}


class RawTool(interfaces.Tool):
  """A tool with a hand-written OpenAPI schema.

  litert_lm.tools derives schemas from Python signatures, which cannot express
  the description-only tools this finetune was trained against.
  """

  def __init__(self, spec: dict[str, Any]) -> None:
    self._spec = spec

  def get_tool_description(self) -> dict[str, Any]:
    return self._spec

  def execute(self, param: Any) -> str:
    return "ok"


def make_tool(name: str) -> RawTool:
  description, props = TOOL_SPECS[name]
  properties = {
      key: {"type": "string", "description": text} for key, text in props.items()
  }
  return RawTool({
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": properties,
              "required": list(props),
          },
      },
  })


def first_tool_call(response: Any) -> tuple[str | None, dict[str, Any], str]:
  """Returns (tool name, arguments, prose) from a non-auto tool-calling reply."""
  calls = response.get("tool_calls") or []
  if calls:
    function = calls[0].get("function", {})
    args = function.get("arguments") or {}
    if isinstance(args, str):
      try:
        args = json.loads(args)
      except json.JSONDecodeError:
        args = {"_unparsed": args}
    if not isinstance(args, dict):
      args = {"_unexpected": args}
    return function.get("name"), args, ""
  prose = ""
  for part in response.get("content") or []:
    prose += part.get("text", "")
  return None, {}, prose.strip()


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
  for key, value in expected.items():
    got = actual.get(key)
    if got is None or str(got).strip().lower() != str(value).strip().lower():
      return False
  return True


def run_set(
    engine: Any, label: str, tool_names: list[str]
) -> dict[str, Any]:
  tools = [make_tool(name) for name in tool_names]
  rows = []
  for prompt, want_tool, want_args in CASES:
    with engine.create_conversation(
        tools=tools, automatic_tool_calling=False
    ) as convo:
      start = time.monotonic()
      response = convo.send_message(prompt)
      elapsed = time.monotonic() - start

    got_tool, got_args, prose = first_tool_call(response)
    tool_ok = got_tool == want_tool
    row = {
        "prompt": prompt,
        "expected_tool": want_tool,
        "expected_args": want_args,
        "got_tool": got_tool,
        "got_args": got_args,
        "tool_match": tool_ok,
        "args_match": tool_ok and args_match(want_args, got_args),
        "prose": prose,
        "seconds": round(elapsed, 3),
    }
    rows.append(row)
    print(
        f"  {'OK ' if tool_ok else 'BAD'} {prompt!r:44} "
        f"want={want_tool:16} got={got_tool}"
    )

  # Only score cases whose expected tool is actually offered.
  offered = set(tool_names)
  scored = [r for r in rows if r["expected_tool"] in offered]
  tool_hits = sum(r["tool_match"] for r in scored)
  full_hits = sum(r["tool_match"] and r["args_match"] for r in scored)
  refusals = sum(1 for r in rows if r["got_tool"] is None)
  latencies = [r["seconds"] for r in rows]
  return {
      "tool_set": label,
      "tools": tool_names,
      "prompts_scored": len(scored),
      "prompts_total": len(rows),
      "tool_name_accuracy": round(tool_hits / len(scored), 4),
      "exact_accuracy": round(full_hits / len(scored), 4),
      "refusal_rate": round(refusals / len(rows), 4),
      "median_seconds": round(statistics.median(latencies), 3),
      "rows": rows,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default=MODEL)
  parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
  parser.add_argument("--out", default="measurements/tool_scaling.json")
  args = parser.parse_args()

  backend: Any = (
      interfaces.Backend.GPU()
      if args.backend == "gpu"
      else interfaces.Backend.CPU()
  )
  engine = litert_lm.Engine(
      model_path=args.model,
      backend=backend,
      max_num_tokens=2048,
      cache_dir=os.path.abspath(".litertlm-cache"),
  )

  results = []
  for label, names in TOOL_SETS.items():
    print(f"\n== {label}: {', '.join(names)}")
    results.append(run_set(engine, label, names))

  summary = [
      {
          "tool_set": r["tool_set"],
          "n_tools": len(r["tools"]),
          "prompts_scored": r["prompts_scored"],
          "tool_name_accuracy": r["tool_name_accuracy"],
          "exact_accuracy": r["exact_accuracy"],
          "refusal_rate": r["refusal_rate"],
          "median_seconds": r["median_seconds"],
      }
      for r in results
  ]

  report = {
      "model": os.path.basename(args.model),
      "backend": args.backend,
      "cases": len(CASES),
      "summary": summary,
      "detail": results,
  }
  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as handle:
    json.dump(report, handle, indent=2)

  print("\n| tool set | n | scored | tool acc | exact | refusal | median s |")
  print("|---|---|---|---|---|---|---|")
  for row in summary:
    print(
        f"| {row['tool_set']} | {row['n_tools']} | {row['prompts_scored']} | "
        f"{row['tool_name_accuracy']:.0%} | {row['exact_accuracy']:.0%} | "
        f"{row['refusal_rate']:.0%} | {row['median_seconds']:.2f} |"
    )
  print(f"\nwrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
