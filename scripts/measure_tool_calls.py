"""Measures FunctionGemma 270M tool-call accuracy and latency via the LiteRT-LM engine.

Run: venv/bin/python scripts/measure_tool_calls.py [--backend cpu] [--out measurements/tool_calls.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import litert_lm
from litert_lm import interfaces

MODEL = "assets/function-gemma-q8-ekv1024.litertlm"

SYSTEM_INSTRUCTION = (
    "You are a mobile assistant on an Android phone. You control the phone by "
    "calling exactly one function. Never reply with prose. If no function "
    "fits, call noop."
)


def open_flashlight() -> str:
  """Turns the phone flashlight on."""
  return "flashlight on"


def close_flashlight() -> str:
  """Turns the phone flashlight off."""
  return "flashlight off"


def append_note(title: str) -> str:
  """Appends a note with the given title to the notes file.

  Args:
    title: The title of the note to write.
  """
  return f"noted: {title}"


def set_alarm(hhmm: str) -> str:
  """Sets an alarm.

  Args:
    hhmm: 24h alarm time as HH:MM.
  """
  return f"alarm set for {hhmm}"


def take_photo() -> str:
  """Takes a photo with the camera."""
  return "photo taken"


def send_message(recipient: str, body: str) -> str:
  """Sends an SMS message.

  Args:
    recipient: Phone number or contact name to message.
    body: The text of the message.
  """
  return f"message sent to {recipient}"


def query_calendar() -> str:
  """Lists today's calendar events."""
  return "no events today"


def noop() -> str:
  """Does nothing."""
  return "noop"


TOOLS = [
    open_flashlight,
    close_flashlight,
    append_note,
    set_alarm,
    take_photo,
    send_message,
    query_calendar,
    noop,
]

# prompt -> (expected tool name, expected argument subset)
CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("turn on the flashlight", "open_flashlight", {}),
    ("switch the flashlight on please", "open_flashlight", {}),
    ("turn off the flashlight", "close_flashlight", {}),
    ("kill the torch", "close_flashlight", {}),
    ("take a photo", "take_photo", {}),
    ("snap a picture", "take_photo", {}),
    ("set an alarm for 07:30", "set_alarm", {"hhmm": "07:30"}),
    ("wake me up at 6am", "set_alarm", {"hhmm": "06:00"}),
    ("what is on my calendar today", "query_calendar", {}),
    ("do i have meetings today", "query_calendar", {}),
    ("send a message to mom saying hi", "send_message", {"recipient": "mom"}),
    ("text alex that i am running late", "send_message", {"recipient": "alex"}),
    ("write a note titled groceries", "append_note", {"title": "groceries"}),
    ("remind me to buy milk", "append_note", {"title": "milk"}),
    ("what is the weather in nairobi", "noop", {}),
    ("thanks, that was helpful", "noop", {}),
]


class RecordingHandler(interfaces.ToolEventHandler):
  """Records tool calls instead of executing them."""

  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
    self.calls.append(tool_call)
    return True

  def process_tool_response(self, tool_response: dict[str, Any]) -> dict[str, Any]:
    return tool_response


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
  for key, value in expected.items():
    got = actual.get(key)
    if got is None:
      return False
    if str(got).strip().lower() != str(value).strip().lower():
      return False
  return True


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
  parser.add_argument("--model", default=MODEL)
  parser.add_argument("--out", default="measurements/tool_calls.json")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  if not os.path.exists(args.model):
    print(f"model not found: {args.model}", file=sys.stderr)
    return 1

  backend: Any
  if args.backend == "gpu":
    backend = interfaces.Backend.GPU()
  else:
    backend = interfaces.Backend.CPU()

  engine = litert_lm.Engine(
      model_path=args.model,
      backend=backend,
      max_num_tokens=2048,
      cache_dir=os.path.abspath(".litertlm-cache"),
  )

  rows = []
  for prompt, want_tool, want_args in CASES:
    handler = RecordingHandler()
    init = time.monotonic()
    with engine.create_conversation(
        messages=[{
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}],
        }],
        tools=TOOLS,
        tool_event_handler=handler,
        automatic_tool_calling=False,
        sampler_config=interfaces.SamplerConfig(temperature=0.0, seed=args.seed),
        max_output_tokens=128,
    ) as convo:
      setup = time.monotonic() - init
      start = time.monotonic()
      response = convo.send_message(prompt)
      elapsed = time.monotonic() - start
      token_count = convo.token_count

    text = ""
    for part in response.get("content", []):
      text += part.get("text", "")

    # automatic_tool_calling=False returns the call in the response rather than
    # routing it through the handler, so read it from there.
    tool_calls = response.get("tool_calls") or []
    call = tool_calls[0].get("function", {}) if tool_calls else {}
    got_tool = call.get("name")
    got_args = call.get("arguments") or {}
    if isinstance(got_args, str):
      try:
        got_args = json.loads(got_args)
      except json.JSONDecodeError:
        got_args = {"_unparsed": got_args}
    if not isinstance(got_args, dict):
      got_args = {"_unexpected": got_args}

    tool_ok = got_tool == want_tool
    args_ok = args_match(want_args, got_args)
    row = {
        "prompt": prompt,
        "expected_tool": want_tool,
        "expected_args": want_args,
        "got_tool": got_tool,
        "got_args": got_args,
        "tool_match": tool_ok,
        "args_match": args_ok,
        "prose": text.strip(),
        "seconds": round(elapsed, 3),
        "kv_tokens": token_count,
    }
    rows.append(row)
    print(
        f"{'OK ' if tool_ok and args_ok else 'BAD'} {prompt!r:52} "
        f"want={want_tool}{want_args or ''} got={got_tool}{got_args or ''}"
    )

  tool_hits = sum(r["tool_match"] for r in rows)
  full_hits = sum(r["tool_match"] and r["args_match"] for r in rows)
  latencies = [r["seconds"] for r in rows]
  report = {
      "model": os.path.basename(args.model),
      "backend": args.backend,
      "runtime": "litert-lm (see pip freeze)",
      "prompt_count": len(rows),
      "tool_name_accuracy": round(tool_hits / len(rows), 4),
      "exact_accuracy": round(full_hits / len(rows), 4),
      "latency_seconds": {
          "median": round(statistics.median(latencies), 3),
          "mean": round(statistics.mean(latencies), 3),
          "max": round(max(latencies), 3),
      },
      "rows": rows,
  }

  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as handle:
    json.dump(report, handle, indent=2)
  print()
  print(f"tool name accuracy: {report['tool_name_accuracy']:.0%}")
  print(f"exact accuracy:     {report['exact_accuracy']:.0%}")
  print(f"median latency:     {report['latency_seconds']['median']:.2f}s")
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
