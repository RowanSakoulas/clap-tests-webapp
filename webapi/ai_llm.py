from __future__ import annotations
import os, json, math, re, statistics as stats
from typing import Any, Dict, List, Optional
from feedback import _spectral_label, _sf
from openai import OpenAI
from pathlib import Path

# ----- helpers -----
_KEY_FILES = [Path("openai_key.txt"), Path(".openai_key")]

def _get_api_key() -> str | None:
    k = os.environ.get("OPENAI_API_KEY")
    if k:
        return k.strip()
    for p in _KEY_FILES:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    return None

def _fin(v): 
    try: return math.isfinite(float(v))
    except: return False

def _median(xs):
    xs=[float(x) for x in xs if _fin(x)]
    return (stats.median(xs) if xs else None)

def _sd(xs):
    xs=[float(x) for x in xs if _fin(x)]
    if len(xs)>=2: 
        try: return stats.pstdev(xs)
        except: return None
    return 0.0 if len(xs)==1 else None

def _choose_rt60(r):
    for k in ("rt60_fused","rt60_measured","rt60_predicted"):
        v=r.get(k)
        if _fin(v): return float(v)
    return None

def _col(results, key):
    out=[]
    for r in results:
        v=r.get(key)
        if _fin(v): out.append(float(v))
    return out

# ----- error helper -----
def _classify_exc(e: Exception) -> tuple[str, str]:
    s = repr(e).lower()
    if "insufficient_quota" in s or "quota" in s:
        return ("insufficient_quota", "AI summary unavailable: insufficient quota")
    if "rate limit" in s or "429" in s:
        return ("rate_limited", "AI summary unavailable: rate limited")
    if "timeout" in s:
        return ("timeout", "AI summary unavailable: timeout")
    return ("llm_call_failed", "AI summary unavailable: provider error")

# ----- facts packing (unchanged logic) -----
def build_facts(results: List[Dict[str,Any]], context: Dict[str,str], ml_info: Dict[str,Any] | None) -> Dict[str,Any]:
    n=len(results)
    counts={"OK":0,"CHECK":0,"LOW":0}
    for r in results:
        q=str(r.get("quality_pred","")).upper()
        if q in counts: counts[q]+=1

    rt60s=[x for x in map(_choose_rt60, results) if x is not None]
    edts =_col(results,"edt_s")
    c50s =_col(results,"c50_db")
    c80s =_col(results,"c80_db")
    l90s =_col(results,"spl_l90_db")
    lows =_col(results,"rt60_low_med")
    highs=_col(results,"rt60_high_med")

    facts = {
        "n_takes": n,
        "counts": counts,
        "context": {
            "use": (context.get("use") or "").strip(),
            "goal": (context.get("goal") or "").strip(),
            "session": (context.get("session") or "").strip(),
        },
        "stats": {
            "rt60_median": _median(rt60s),
            "rt60_sd": _sd(rt60s),
            "edt_median": _median(edts),
            "c50_median": _median(c50s),
            "c80_median": _median(c80s),
            "l90_median": _median(l90s),
            "low_median": _median(lows),
            "high_median": _median(highs),
        },
        "flags": {},
        "ml": ml_info or {}
    }

    rt60_med = facts["stats"]["rt60_median"]
    facts["flags"]["rt60_above_target"] = False
    facts["flags"]["rt60_below_target"] = False

    low_med  = facts["stats"]["low_median"]
    high_med = facts["stats"]["high_median"]
    spectral = _spectral_label(low_med if low_med is not None else float("nan"),
                               high_med if high_med is not None else float("nan"))
    facts["flags"]["spectral_label"] = spectral

    def _is_high_band(v):
        if not _fin(v):
            return False
        return float(v) > 1.80
    facts["flags"]["low_band_high_majority"]  = (sum(_is_high_band(r.get("rt60_low_med"))  for r in results) >= max(1, n//2))
    facts["flags"]["high_band_high_majority"] = (sum(_is_high_band(r.get("rt60_high_med")) for r in results) >= max(1, n//2))

    capped_fraction = (sum(1 for r in results if (r.get("qc",{}) or {}).get("capped"))/n) if n else 0.0
    facts["stats"]["capped_fraction"] = capped_fraction
    facts["flags"]["many_capped"] = capped_fraction >= 0.25

    return facts

# ----- prompts -----
SYSTEM_PROMPT = """You are an acoustics assistant for a general audience. Each JSON you receive describes a single room session made up of multiple recordings in the same space.

Your job is to summarise how the room behaves acoustically and suggest practical ways to improve it, using Australian English.

Overall style
- 3 or 4 short bullets, then 1 - 2 action bullets.
- Each bullet is one sentence, no run-ons.
- Vary sentence openings so the text does not feel like template blanks.
- Sound neutral and encouraging; suggestions are guidance, not strict rules.

Numbers and metrics
- Use parentheses with equals for numbers, for example (RT60 = 0.45 s), (C50 = -2.50 dB), (L90 = 38.0 dB).
- Only use the numbers and boolean flags provided in the JSON facts; do not invent new values or thresholds.
- Do not assume fixed “ideal” RT60 ranges for all rooms. You may describe RT60 qualitatively as short, moderate, or long based on the value itself, but do not refer to any hidden target band.

Interpretation hints
- Treat C50 as a speech intelligibility indicator and C80 as a music / definition clarity indicator. You may describe them qualitatively (for example “speech clarity is limited” or “music clarity is strong”), using the given values.
- Treat L90 as background sound level; a higher L90 means more constant background noise.
- When many_capped is true, interpret the session as inconclusive: too many decay estimates hit the recording length. The priority is to recommend retesting using the recording guide, rather than drawing strong conclusions.

Treatment and actions
- When giving actions, always link them to at least one key metric or pattern mentioned in the bullets (for example overall RT60 length, low-frequency emphasis, brightness, clarity, or background noise).
- It is fine to mention generic acoustic treatment types such as:
  - added soft furnishings (curtains, rugs, upholstered furniture),
  - wall or ceiling absorption panels,
  - bass control in corners (for example thicker absorptive elements or bass traps),
  - diffusing elements that break up flat walls (for example uneven furniture or shelving).
- Make it clear that these are examples of ways to break up reflections, not mandatory items.
- Do not mention specific product names, brands, or detailed construction specifications.
- Keep the overall focus on the acoustic performance of the room and how sound behaves, not on interior styling."""

USER_TEMPLATE = """Facts for one room session (JSON):
{facts_json}

You are summarising this single room session, which may contain several recordings in the same space.

Write a JSON object with exactly two arrays: "bullets" and "actions".
The "bullets" array must contain only the 3-4 descriptive bullets.
The "actions" array must contain only 1-2 action suggestions.
Do NOT put any actions inside the "bullets" array.
Do NOT write the word "Actions:" anywhere in any bullet or action text.

Write 3 or 4 bullets in this fixed order:

Bullet 1 - Context and overall RT:
- If context.use or context.goal exist, briefly mention how the room is used and/or what the user wants (for example quiet, lively, clear).
- Always mention the median RT60 from stats.rt60_median, in the format (RT60 = x.xx s).
- Without using any target band, you may describe the decay qualitatively (for example “quite short”, “moderate”, or “rather long”) based on the number itself.

Bullet 2 - Snapshot and reliability:
- Use n_takes and counts.OK / counts.CHECK / counts.LOW to describe how much data the session is based on and whether most takes are usable.
- If flags.many_capped is true, clearly state that many estimates reached the recording length, the session should be treated as a retest case, and new recordings should be made following the recording guide, rather than over-interpreting the current numbers.

Bullet 3 - Decay character and spectrum:
- Use flags.spectral_label and the low/high band majority flags to describe decay and spectrum.
- If spectral_label is "boomy" or low_band_high_majority is true, say that low frequencies are strong and the room may feel bass-heavy.
- If spectral_label is "bright" or high_band_high_majority is true, say that high frequencies dominate and the room may feel bright.
- If spectral_label is "balanced" and neither majority flag is true, say that spectral balance is roughly even.
- Keep the language soft and descriptive (“tends to feel…”, “leans toward…”) rather than absolute.

Bullet 4 - Clarity and background noise (include this bullet when enough data is present):
- If stats.c50_median is present, mention it explicitly as a speech clarity measure using (C50 = x.xx dB) and describe speech intelligibility qualitatively.
- If stats.c80_median is present, mention it explicitly as a music / definition clarity measure using (C80 = x.xx dB) and describe musical clarity qualitatively.
- If stats.l90_median is present, mention background sound using (L90 = x.xx dB) and describe whether the background noise level is low, moderate, or high in plain language (remember that higher L90 means more constant background noise).

Actions (1 - 2 items):
- Each action must clearly relate to at least one point from the bullets
  (for example long decay, dominant low end, bright top end, limited clarity, or high
  background noise).
- Actions should also refer to the room use or goal when they can, for example
  “For a bedroom aiming to feel calm and clear, to improve acoustic quality you could…”.
- Example patterns:
  - If RT60 is noticeably long or low frequencies dominate, suggest adding bass control in
    corners and extra wall or ceiling absorption to bring the RT60 closer to what would be
    expected for that room type, plus more furniture or surfaces that break up flat reflections.
  - If the room is bright, suggest adding higher-frequency absorption or diffusion near reflection
    points (for example wall panels, curtains, rugs, shelving or uneven furniture that breaks up
    the wall).
  - If clarity metrics are weak while RT60 is not extreme, suggest soft furnishings or panels
    around the main listening or speaking area to enhance clarity and keep voices intelligible.
  - If L90 is high, suggest reducing constant noise sources or improving isolation where practical
    to reduce distractions from background noise, but avoid over-promising.
- Even when the room is broadly suitable, provide at least one optional improvement
  (“If you wanted to improve the acoustic quality of the room further, one option is to add…”).

Return strict JSON only, with this exact shape and double quotes:

{{"bullets": ["..."], "actions": ["..."]}}"""

def _split_actions_text(text: str) -> List[str]:
    """Turn a single 'Actions: ...' string into clean action sentences."""
    if not isinstance(text, str):
        text = str(text)
    t = text.strip()
    if not t:
        return []

    # Remove leading 'Actions:' if present
    t = re.sub(r'^actions:\s*', '', t, flags=re.I).strip()
    if not t:
        return []

    # Normalise odd joins like "., " and ",." into a normal sentence break
    t = re.sub(r'\.\s*,\s*', '. ', t)
    t = re.sub(r',\s*\.\s*', '. ', t)

    # Split on sentence boundaries: punctuation + space + capital letter
    parts = re.split(r'(?<=[.?!])\s+(?=[A-Z])', t)

    out: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Make sure each part ends with punctuation
        if not re.search(r'[.?!]$', p):
            p += "."
        out.append(p)
    return out

def _normalise_summary(bullets_raw, actions_raw) -> tuple[List[str], List[str]]:
    """
    Ensure:
    - bullets do NOT contain 'Actions:' lines
    - actions is a flat list of 1-3 clean sentences
    """
    bullets_raw = list(bullets_raw or [])
    actions_raw = list(actions_raw or [])

    bullets_clean: List[str] = []
    actions: List[str] = []

    # Pull any 'Actions:' bullets out into the actions list
    for b in bullets_raw:
        s = str(b)
        if re.match(r'\s*actions:\s*', s, flags=re.I):
            actions.extend(_split_actions_text(s))
        else:
            bullets_clean.append(s)

    # Also normalise anything already in the actions array
    for a in actions_raw:
        actions.extend(_split_actions_text(a))

    # De-duplicate actions while preserving order
    seen = set()
    uniq_actions: List[str] = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            uniq_actions.append(a)

    return bullets_clean[:5], uniq_actions[:3]

# ----- OpenAI call -----
def _call_openai(model: str, prompt_sys: str, prompt_user: str) -> str:
    client = OpenAI(api_key=_get_api_key())
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":prompt_sys},
                  {"role":"user","content":prompt_user}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ----- main entry (OpenAI only) -----
def generate_summary(results, context, ml_info=None):
    facts = build_facts(results, context, ml_info)

    api_key = _get_api_key()
    if not api_key:
        return {"bullets": [], "actions": [], "error": "provider_disabled", "error_msg": "AI summary disabled"}

    model = os.environ.get("AI_MODEL") or "gpt-4o-mini"
    facts_json = json.dumps(facts, ensure_ascii=False, default=lambda x: None)
    user_prompt = USER_TEMPLATE.format(facts_json=facts_json)
    print(f"[AI] provider=openai model={model} key_len={len(api_key)}")

    try:
        raw = _call_openai(model, SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        kind, msg = _classify_exc(e)
        print("[AI] call failed:", repr(e))
        return {"bullets": [], "actions": [], "error": kind, "error_msg": msg}

    try:
        start = raw.find("{"); end = raw.rfind("}")
        data = json.loads(raw[start:end+1])

        bullets_raw = data.get("bullets", [])
        actions_raw = data.get("actions", [])

        # Canonicalise: strip 'Actions:' bullets out and split into proper actions
        bullets, actions = _normalise_summary(bullets_raw, actions_raw)

        return {"bullets": bullets, "actions": actions}
    except Exception:
        return {
            "bullets": [],
            "actions": [],
            "error": "parse_failed",
            "error_msg": "AI summary unavailable: parse error"
        }
