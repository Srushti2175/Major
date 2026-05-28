"""
Audio Alert Mapping — maps YAMNet class names to dashboard alert levels.

Three levels
------------
- **alert**     → Screaming / distress / alarms / dangerous sounds / SILENCE
- **attention** → Someone talking / speech detected
- **normal**    → Everything else (background sounds: music, water, TV, etc.)

Logic priority:  alert  >  attention  >  normal
"""

from __future__ import annotations


# ------------------------------------------------------------------ #
# 🔴  ALERT — immediate danger, distress, or suspicious silence       #
# ------------------------------------------------------------------ #

ALERT_CLASSES: set[str] = {
    # Screaming / shouting / distress vocalisation
    "Screaming",
    "Shout",
    "Yell",
    "Battle cry",
    "Children shouting",
    "Crying, sobbing",
    "Whimper",
    "Wail, moan",
    "Groan",
    "Gasp",
    "Pant",
    "Choking",

    # Alarms & sirens
    "Fire alarm",
    "Smoke detector, smoke alarm",
    "Siren",
    "Alarm",
    "Buzzer",
    "Civil defense siren",
    "Alarm clock",

    # Impact / fall / danger sounds
    "Explosion",
    "Gunshot, gunfire",
    "Glass",          # breaking glass
    "Crash",
    "Thump, thud",    # fall impact
    "Bang",

    # Awkward / suspicious silence
    # YAMNet labels these when no meaningful audio is detected
    "Silence",
    "White noise",
    "Inside, small room",   # sometimes indicates empty/silent room
}


# ------------------------------------------------------------------ #
# 🟡  ATTENTION — someone is talking; may need a check               #
# ------------------------------------------------------------------ #

ATTENTION_CLASSES: set[str] = {
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Whispering",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
}


# ------------------------------------------------------------------ #
# Public mapping function                                              #
# ------------------------------------------------------------------ #

def map_yamnet_to_level(yamnet_label: str) -> str:
    """
    Map a single YAMNet class name to one of three alert levels.

    Priority:  alert > attention > normal

    Parameters
    ----------
    yamnet_label : str
        The ``top_class`` string returned by
        :func:`audio_yamnet.predict_audio_events`.

    Returns
    -------
    str
        ``"alert"`` | ``"attention"`` | ``"normal"``
    """
    if yamnet_label in ALERT_CLASSES:
        return "alert"
    if yamnet_label in ATTENTION_CLASSES:
        return "attention"
    return "normal"


def map_events_to_level(event_list: list[str]) -> str:
    """
    Given a list of detected event names, return the *highest* severity
    level found across all events.

    Priority:  alert > attention > normal

    Parameters
    ----------
    event_list : list[str]
        List of YAMNet class names that scored above the threshold.

    Returns
    -------
    str
        ``"alert"`` | ``"attention"`` | ``"normal"``
    """
    has_attention = False
    for event in event_list:
        if event in ALERT_CLASSES:
            return "alert"          # short-circuit — can't get worse
        if event in ATTENTION_CLASSES:
            has_attention = True
    return "attention" if has_attention else "normal"
