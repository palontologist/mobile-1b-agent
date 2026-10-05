"""Tests whether FunctionGemma 270M routes on semantics or on literal keywords,
and how sensitive routing is to the tool description's wording.

Both use the 4-tool set that measure_tool_scaling.py found to score highest.

Run: venv/bin/python scripts/measure_prompt_sensitivity.py
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

TOOLS_WORDED_AS_FINETUNE: list[tuple[str, str]] = [
    ("open_flashlight", "Turns the phone's flashlight on."),
    ("close_flashlight", "Turns the phone's flashlight off."),
    ("query_calendar", "Lists the events on the user's calendar for today."),
    ("take_photo", "Takes a photo with the phone's camera."),
]

TOOLS_REWORDED: list[tuple[str, str]] = [
    ("open_flashlight", "Enables the LED torch."),
    ("close_flashlight", "Disables the LED torch."),
    ("query_calendar", "Shows the user's schedule."),
    ("take_photo", "Captures a still image."),
]

# Each intent appears once as a prompt containing the intent's own vocabulary
# and once, or more, that avoids it. A model routing on meaning should score the
# same on both; a model matching keywords should not.
PARAPHRASE_PROBES: list[tuple[str, str, str]] = [
    ("what is on my calendar today", "query_calendar", "literal"),
    ("whats on my calender today", "query_calendar", "paraphrase"),
    ("calendar today", "query_calendar", "terse"),
    ("what's my schedule", "query_calendar", "paraphrase"),
    ("list my calendar events", "query_calendar", "paraphrase"),
    ("do i have meetings today", "query_calendar", "paraphrase"),
    ("am i busy today", "query_calendar", "paraphrase"),
    ("show my events", "query_calendar", "paraphrase"),
    ("turn on the flashlight", "open_flashlight", "literal"),
    ("i need some light in here", "open_flashlight", "paraphrase"),
    ("it is dark in here", "open_flashlight", "paraphrase"),
    ("lights on please", "open_flashlight", "paraphrase"),
    ("turn off the flashlight", "close_flashlight", "literal"),
    ("kill the torch now", "close_flashlight", "paraphrase"),
    ("shut the light", "close_flashlight", "paraphrase"),
    ("no more light", "close_flashlight", "paraphrase"),
]


class RawTool(interfaces.Tool):
  """A tool with a hand-written OpenAPI schema."""

  def __init__(self, spec: dict[str, Any]) -> None:
    self._spec = spec

  def get_tool_description(self) -> dict[str, Any]:
    return self._spec

  def execute(self, param: Any) -> str:
    return "ok"


def make_tools(pairs: list[tuple[str, str]]) -> list[RawTool]:
  return [
      RawTool({
          "type": "function",
          "function": {
              "name": name,
              "description": description,
              "parameters": {
                  "type": "object",
                  "properties": {},
                  "required": [],
              },
          },
      })
      for name, description in pairs
  ]


def ask(engine: Any, tools: list[RawTool], prompt: str) -> dict[str, Any]:
  with engine.create_conversation(
      tools=tools, automatic_tool_calling=False
  ) as convo:
    start = time.monotonic()
    response = convo.send_message(prompt)
    elapsed = time.monotonic() - start
  calls = response.get("tool_calls") or []
  if calls:
    got = calls[0].get("function", {}).get("name")
  else:
    got = None
  prose = "".join(p.get("text", "") for p in response.get("content") or [])
  return {
      "got_tool": got,
      "refusal": got is None,
      "prose": prose.strip(),
      "seconds": round(elapsed, 3),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default=MODEL)
  parser.add_argument("--out", default="measurements/prompt_sensitivity.json")
  args = parser.parse_args()

  engine = litert_lm.Engine(
      model_path=args.model,
      backend=interfaces.Backend.CPU(),
      max_num_tokens=2048,
      cache_dir=os.path.abspath(".litertlm-cache"),
  )

  worded = make_tools(TOOLS_WORDED_AS_FINETUNE)
  reworded = make_tools(TOOLS_REWORDED)

  print("== keyword-literal vs paraphrase (4 tools, finetune wording)")
  probes = []
  for prompt, expected, kind in PARAPHRASE_PROBES:
    result = ask(engine, worded, prompt)
    probes.append({
        "prompt": prompt,
        "expected_tool": expected,
        "kind": kind,
        "tool_match": result["got_tool"] == expected,
        "refusal": result["refusal"],
        "prose": result["prose"],
        "seconds": result["seconds"],
    })
    print(
        f"  {'OK ' if result['got_tool'] == expected else 'BAD'} "
        f"{kind:11} {prompt!r:38} -> {result['got_tool'] or 'refusal'}"
    )

  by_kind: dict[str, list[bool]] = {}
  for probe in probes:
    by_kind.setdefault(probe["kind"], []).append(probe["tool_match"])
  kind_summary = {
      kind: {
          "n": len(hits),
          "accuracy": round(sum(hits) / len(hits), 4),
      }
      for kind, hits in sorted(by_kind.items())
  }

  print("\n== description-wording sensitivity (same prompts, reworded tools)")
  wording = []
  for prompt, expected, kind in PARAPHRASE_PROBES:
    before = ask(engine, worded, prompt)
    after = ask(engine, reworded, prompt)
    row = {
        "prompt": prompt,
        "expected_tool": expected,
        "kind": kind,
        "match_worded": before["got_tool"] == expected,
        "match_reworded": after["got_tool"] == expected,
    }
    wording.append(row)
    print(
        f"  {prompt!r:38} worded={before['got_tool'] or 'refusal':16} "
        f"reworded={after['got_tool'] or 'refusal'}"
    )

  flips = [r for r in wording if r["match_worded"] != r["match_reworded"]]
  latencies = [r["seconds"] for r in probes]

  report = {
      "model": os.path.basename(args.model),
      "backend": "cpu",
      "paraphrase": {
          "by_kind": kind_summary,
          "probes": probes,
      },
      "description_wording": {
          "n_prompts": len(wording),
          "n_flipped": len(flips),
          "rows": wording,
      },
      "median_seconds": round(statistics.median(latencies), 3),
  }
  os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
  with open(args.out, "w") as handle:
    json.dump(report, handle, indent=2)

  print("\n| prompt kind | n | routing accuracy |")
  print("|---|---|---|")
  for kind, stats in kind_summary.items():
    print(f"| {kind} | {stats['n']} | {stats['accuracy']:.0%} |")
  print(
      f"\nreworded the tool descriptions and {len(flips)}/{len(wording)} "
      f"prompts changed verdict"
  )
  print(f"wrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
