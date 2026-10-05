# Held-out evaluation set for the tool router.
#
# RULES FOR THIS FILE (follow them, or the result is not held-out):
#  1. No string here may appear in router_v2.POSITIVE or router_v2.NEGATIVES.
#  2. Phrasings must not be written by iterating on the prototypes. Registers,
#     sentence shapes and failure modes should be ones the prototype bank does
#     not already anticipate.
#  3. Every label is a deliberate commitment made BEFORE the run. Ambiguous cases
#     are labelled `clarify` and count against false-acts, not as misses.
#  4. Negated requests are included on purpose. v2 has no negation handling and
#     these are expected to fail; they are here so the failure is on record
#     rather than absent.
#
# Written 2026-10-05, after router v2 existed but before it was ever run on this
# file. 42 cases.

# (utterance, expected, bucket)
# expected is one of: open_flashlight, close_flashlight, query_calendar,
#                     take_photo, clarify
HELD_OUT = [
    # --- indirect / situational: intent stated, never named --------------------
    ("my hands are full and I can't see what I'm doing", "open_flashlight", "indirect"),
    ("it's pitch black in here", "open_flashlight", "indirect"),
    ("the room is too dark to read anything", "open_flashlight", "indirect"),
    ("I keep knocking into things in here", "open_flashlight", "indirect"),
    ("I don't need it any more, and it's blinding me", "close_flashlight", "indirect"),
    ("that beam is giving me a headache", "close_flashlight", "indirect"),
    ("the battery is suffering because of that lamp", "close_flashlight", "indirect"),

    # --- polite / verbose framing ---------------------------------------------
    ("could you possibly turn the torch on for me", "open_flashlight", "polite"),
    ("would you mind switching the flashlight off", "close_flashlight", "polite"),
    ("when you get a second, could you snap a quick photo", "take_photo", "polite"),
    ("I was wondering if you could tell me what I've got lined up today", "query_calendar", "polite"),
    ("hey, quick one before I forget - what's on today", "query_calendar", "polite"),

    # --- bare keywords / fragments -------------------------------------------
    ("torch", "open_flashlight", "fragment"),
    ("flashlight off", "close_flashlight", "fragment"),
    ("today's agenda", "query_calendar", "fragment"),
    ("pic", "take_photo", "fragment"),

    # --- NEGATION: expected to be clarify. v2 has no negation handling. --------
    ("don't turn on the light", "clarify", "negation"),
    ("do not switch the torch on", "clarify", "negation"),
    ("never turn the flashlight off", "clarify", "negation"),
    ("I don't want to take a photo", "clarify", "negation"),
    ("don't show me my calendar", "clarify", "negation"),

    # --- antonym pairs on the confusable pair ---------------------------------
    ("switch the torch on", "open_flashlight", "antonym"),
    ("switch the torch off", "close_flashlight", "antonym"),
    ("power the torch on", "open_flashlight", "antonym"),
    ("power the torch off", "close_flashlight", "antonym"),
    ("the torch is on now", "open_flashlight", "antonym"),
    ("the torch is off now", "close_flashlight", "antonym"),

    # --- typos: not just transpositions ---------------------------------------
    ("turn onn the flashlught", "open_flashlight", "typo"),
    ("tirn of the flaslight", "close_flashlight", "typo"),
    ("whats on my calndar", "query_calendar", "typo"),
    ("tka a snap", "take_photo", "typo"),

    # --- multi-intent ----------------------------------------------------------
    ("switch the torch on then tell me what's on today", "clarify", "multi"),
    ("snap something and then turn the torch off", "clarify", "multi"),

    # --- out of scope ----------------------------------------------------------
    ("how much data have I used this month", "clarify", "out_of_scope"),
    ("call my brother", "clarify", "out_of_scope"),
    ("what's the exchange rate", "clarify", "out_of_scope"),
    ("read my last message aloud", "clarify", "out_of_scope"),
    ("how do I tie a knot", "clarify", "out_of_scope"),

    # --- genuinely ambiguous ---------------------------------------------------
    ("is the torch still on", "clarify", "ambiguous"),
    ("would turning it off be sensible in here", "clarify", "ambiguous"),
    ("same as last time", "clarify", "ambiguous"),

    # --- ordinary scoreable, different register --------------------------------
    ("give us some light in here", "open_flashlight", "plain"),
    ("put the lamp out", "close_flashlight", "plain"),
    ("anything on for me today", "query_calendar", "plain"),
    ("capture what I'm looking at", "take_photo", "plain"),
]