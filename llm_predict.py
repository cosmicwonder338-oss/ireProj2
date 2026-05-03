# import google.generativeai as genai
# import os

# genai.configure(api_key="AIzaSyDhSMYIavrWceskOe6zoEk5_cRH0ogr7kE")

# model = genai.GenerativeModel("gemini-1.5-flash")


# def predict_with_llm(claim, evidence_list):

#     if not evidence_list:
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 100},
#             "used_evidence": "none"
#         }

#     evidence_text = "\n".join(evidence_list[:5])

#     prompt = f"""
# You are a fact verification system.

# Claim:
# {claim}

# Evidence:
# {evidence_text}

# Task:
# Classify the claim into one of:
# - SUPPORTS
# - REFUTES
# - NOT ENOUGH INFO

# Rules:
# - Use ONLY the evidence provided
# - If evidence clearly supports → SUPPORTS
# - If contradicts → REFUTES
# - If insufficient → NOT ENOUGH INFO

# Answer ONLY in this format:
# LABEL: <label>
# """

#     response = model.generate_content(prompt)

#     text = response.text.strip().upper()

#     if "SUPPORTS" in text:
#         label = "SUPPORTS"
#     elif "REFUTES" in text:
#         label = "REFUTES"
#     else:
#         label = "NOT ENOUGH INFO"

#     return {
#         "label": label,
#         "scores": {
#             "SUPPORTS": 0,
#             "REFUTES": 0,
#             "NOT ENOUGH INFO": 0
#         },
#         "used_evidence": evidence_list[0]
#     }  ##  api_key="AIzaSyDhSMYIavrWceskOe6zoEk5_cRH0ogr7kE"


# from google import genai
# import re

# client = genai.Client(api_key="AIzaSyDhSMYIavrWceskOe6zoEk5_cRH0ogr7kE")
# MODEL = "gemini-1.5-flash"


# # --------------------------
# # SAFE NUMBER PARSER
# # --------------------------
# def extract_score(text, label):
#     patterns = [
#         rf"{label}\s*[:=\-]\s*(\d+)",
#         rf"{label}\s*(\d+)"
#     ]

#     for p in patterns:
#         match = re.search(p, text)
#         if match:
#             return float(match.group(1))

#     return 0.0


# # --------------------------
# # SAFE LABEL PARSER
# # --------------------------
# def extract_label(text):
#     match = re.search(r"LABEL\s*[:=\-]\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)", text)
#     if match:
#         return match.group(1)

#     return "NOT ENOUGH INFO"


# # --------------------------
# # MAIN FUNCTION
# # --------------------------
# def predict_with_llm(claim, evidence_list):

#     if not evidence_list:
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {
#                 "SUPPORTS": 0,
#                 "REFUTES": 0,
#                 "NOT ENOUGH INFO": 100
#             }
#         }

#     evidence_text = "\n".join(evidence_list[:5])

#     prompt = f"""
# You are a STRICT fact verification system.

# Claim:
# {claim}

# Evidence:
# {evidence_text}

# Task:
# 1. Decide ONE label: SUPPORTS / REFUTES / NOT ENOUGH INFO
# 2. Assign confidence scores (0–100)
# 3. Scores MUST sum to 100

# Output EXACTLY like this (no extra text):

# LABEL: SUPPORTS
# SUPPORTS: 70
# REFUTES: 10
# NOT ENOUGH INFO: 20
# """

#     try:
#         response = client.models.generate_content(
#             model=MODEL,
#             contents=prompt
#         )

#         text = response.text.upper()
#         print("\n🔍 RAW LLM OUTPUT:\n", text)

#     except Exception as e:
#         print("LLM ERROR:", e)
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {
#                 "SUPPORTS": 33.3,
#                 "REFUTES": 33.3,
#                 "NOT ENOUGH INFO": 33.3
#             }
#         }

#     # --------------------------
#     # PARSE
#     # --------------------------
#     label = extract_label(text)

#     supports = extract_score(text, "SUPPORTS")
#     refutes = extract_score(text, "REFUTES")
#     nei = extract_score(text, "NOT ENOUGH INFO")

#     total = supports + refutes + nei

#     # --------------------------
#     # 🔥 FIX: normalize if messy
#     # --------------------------
#     if total == 0:
#         supports, refutes, nei = 33.3, 33.3, 33.3
#     else:
#         supports = (supports / total) * 100
#         refutes = (refutes / total) * 100
#         nei = (nei / total) * 100

#     return {
#         "label": label,
#         "scores": {
#             "SUPPORTS": supports,
#             "REFUTES": refutes,
#             "NOT ENOUGH INFO": nei
#         }
#     }


# from google import genai
# import os
# import json

# # --------------------------
# # 🔐 SET API KEY (SAFE WAY)
# # --------------------------
# client = genai.Client(
#     api_key="AIzaSyDhSMYIavrWceskOe6zoEk5_cRH0ogr7kE"  # export GOOGLE_API_KEY=your_key
# )

# MODEL = "gemini-1.5-flash-latest"


# # --------------------------
# # MAIN FUNCTION
# # --------------------------
# def predict_with_llm(claim, evidence_list):

#     # --------------------------
#     # NO EVIDENCE CASE
#     # --------------------------
#     if not evidence_list:
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {
#                 "SUPPORTS": 0.0,
#                 "REFUTES": 0.0,
#                 "NOT ENOUGH INFO": 100.0
#             }
#         }

#     evidence_text = "\n".join(evidence_list[:5])

#     # --------------------------
#     # 🔥 STRICT JSON PROMPT
#     # --------------------------
#     prompt = f"""
# You are a STRICT fact verification system.

# Claim:
# {claim}

# Evidence:
# {evidence_text}

# Task:
# 1. Predict label: SUPPORTS / REFUTES / NOT ENOUGH INFO
# 2. Assign confidence scores (0–100)

# Rules:
# - Use ONLY the evidence
# - Scores MUST sum to 100
# - Output MUST be valid JSON
# - NO explanation
# - NO extra text

# Output EXACTLY:

# {{
#   "label": "SUPPORTS",
#   "scores": {{
#     "SUPPORTS": 70,
#     "REFUTES": 10,
#     "NOT ENOUGH INFO": 20
#   }}
# }}
# """

#     try:
#         response = client.models.generate_content(
#             model=MODEL,
#             contents=prompt
#         )

#         text = response.text.strip()
#         print("\n🔍 RAW LLM OUTPUT:\n", text)

#         # --------------------------
#         # 🔥 SAFE JSON EXTRACTION
#         # --------------------------
#         start = text.find("{")
#         end = text.rfind("}") + 1
#         json_str = text[start:end]

#         data = json.loads(json_str)

#         label = data.get("label", "NOT ENOUGH INFO")
#         scores = data.get("scores", {})

#         supports = float(scores.get("SUPPORTS", 0))
#         refutes = float(scores.get("REFUTES", 0))
#         nei = float(scores.get("NOT ENOUGH INFO", 0))

#         # --------------------------
#         # 🔥 NORMALIZE (IMPORTANT)
#         # --------------------------
#         total = supports + refutes + nei

#         if total == 0:
#             supports, refutes, nei = 33.3, 33.3, 33.3
#         else:
#             supports = (supports / total) * 100
#             refutes = (refutes / total) * 100
#             nei = (nei / total) * 100

#         return {
#             "label": label,
#             "scores": {
#                 "SUPPORTS": supports,
#                 "REFUTES": refutes,
#                 "NOT ENOUGH INFO": nei
#             }
#         }

#     except Exception as e:
#         print("❌ LLM ERROR / PARSE ERROR:", e)

#         # --------------------------
#         # FALLBACK
#         # --------------------------
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {
#                 "SUPPORTS": 33.3,
#                 "REFUTES": 33.3,
#                 "NOT ENOUGH INFO": 33.3
#             }
#         }


# from google import genai
# import json
# import re

# client = genai.Client(api_key="AIzaSyDhSMYIavrWceskOe6zoEk5_cRH0ogr7kE")
# MODEL = "gemini-1.5-flash-latest"


# def safe_parse_json(text):
#     try:
#         start = text.find("{")
#         end = text.rfind("}") + 1
#         return json.loads(text[start:end])
#     except:
#         return None


# def predict_with_llm(claim, evidence_list):

#     if not evidence_list:
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {
#                 "SUPPORTS": 0,
#                 "REFUTES": 0,
#                 "NOT ENOUGH INFO": 100
#             }
#         }

#     evidence_text = "\n".join(evidence_list[:5])

#     prompt = f"""
# Return ONLY valid JSON. No explanation.

# Claim: {claim}

# Evidence:
# {evidence_text}

# Output format:
# {{
#   "label": "SUPPORTS | REFUTES | NOT ENOUGH INFO",
#   "scores": {{
#     "SUPPORTS": number,
#     "REFUTES": number,
#     "NOT ENOUGH INFO": number
#   }}
# }}

# Rules:
# - Scores MUST sum to 100
# - Use only evidence
# """

#     # 🔁 TRY 2 TIMES
#     for attempt in range(2):

#         try:
#             response = client.models.generate_content(
#                 model=MODEL,
#                 contents=prompt
#             )

#             text = response.text.strip()
#             print("\n🔍 RAW OUTPUT:\n", text)

#             data = safe_parse_json(text)

#             if data is None:
#                 continue

#             label = data.get("label", "NOT ENOUGH INFO")
#             scores = data.get("scores", {})

#             s = float(scores.get("SUPPORTS", 0))
#             r = float(scores.get("REFUTES", 0))
#             n = float(scores.get("NOT ENOUGH INFO", 0))

#             total = s + r + n

#             # ✅ FIX: normalize EVEN if partial
#             if total > 0:
#                 s = (s / total) * 100
#                 r = (r / total) * 100
#                 n = (n / total) * 100
#             else:
#                 continue  # retry

#             return {
#                 "label": label,
#                 "scores": {
#                     "SUPPORTS": s,
#                     "REFUTES": r,
#                     "NOT ENOUGH INFO": n
#                 }
#             }

#         except Exception as e:
#             print("ERROR:", e)

#     # ❌ FINAL FALLBACK
#     return {
#         "label": "NOT ENOUGH INFO",
#         "scores": {
#             "SUPPORTS": 33.3,
#             "REFUTES": 33.3,
#             "NOT ENOUGH INFO": 33.3
#         }
#     }

# from google import genai
# from google.genai import types
# import json

# # ─────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────
# client = genai.Client(api_key="AIzaSyDhSMYIavrWceskOe6zoEk5_cRH0ogr7kE")
# MODEL = "gemini-1.5-flash-latest"


# # ─────────────────────────────────────────
# # 1. JSON PARSER
# # ─────────────────────────────────────────
# def safe_parse_json(text):
#     """Try multiple strategies to extract JSON from LLM output."""
#     # Remove markdown fences
#     text = text.replace("```json", "").replace("```", "").strip()

#     # Strategy 1: direct parse
#     try:
#         return json.loads(text)
#     except:
#         pass

#     # Strategy 2: find first { to last }
#     try:
#         start = text.find("{")
#         end = text.rfind("}") + 1
#         if start != -1 and end > start:
#             return json.loads(text[start:end])
#     except:
#         pass

#     return None


# # ─────────────────────────────────────────
# # 2. KEYWORD FALLBACK (when LLM fails)
# # ─────────────────────────────────────────
# def _keyword_fallback(claim, evidence_list):
#     """
#     Simple keyword-based scoring when LLM is unavailable.
#     Much better than hardcoded equal scores.
#     """
#     claim_words = set(claim.lower().split())
#     support_hits = 0
#     refute_hits = 0

#     refute_words = {"not", "never", "false", "incorrect", "wrong", "no",
#                     "cannot", "isn't", "aren't", "wasn't", "doesn't", "don't"}
#     support_words = {"is", "was", "are", "confirmed", "true", "correct",
#                      "yes", "indeed", "does", "has", "have", "will"}

#     for evidence in evidence_list[:5]:
#         ev_lower = evidence.lower()
#         ev_words = set(ev_lower.split())
#         overlap = len(claim_words & ev_words) / max(len(claim_words), 1)

#         if overlap > 0.3:
#             if ev_words & refute_words:
#                 refute_hits += 1
#             elif ev_words & support_words:
#                 support_hits += 1

#     total_hits = support_hits + refute_hits

#     if total_hits == 0:
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {"SUPPORTS": 10.0, "REFUTES": 10.0, "NOT ENOUGH INFO": 80.0}
#         }

#     s = round((support_hits / total_hits) * 70, 2)
#     r = round((refute_hits / total_hits) * 70, 2)
#     n = round(100.0 - s - r, 2)

#     score_map = {"SUPPORTS": s, "REFUTES": r, "NOT ENOUGH INFO": n}
#     label = max(score_map, key=lambda k: score_map[k])

#     return {"label": label, "scores": score_map}


# # ─────────────────────────────────────────
# # 3. EVIDENCE FILTER
# # ─────────────────────────────────────────
# def filter_relevant_evidence(claim, evidence_list, min_word_overlap=0.2):
#     """Remove evidence that shares almost no words with the claim."""
#     claim_words = set(claim.lower().split())
#     filtered = []

#     for ev in evidence_list:
#         ev_words = set(ev.lower().split())
#         overlap = len(claim_words & ev_words) / max(len(claim_words), 1)
#         if overlap >= min_word_overlap:
#             filtered.append(ev)

#     print(f"🔍 Evidence filter: {len(evidence_list)} → {len(filtered)} relevant")

#     # If nothing passes filter, return all (better than empty)
#     return filtered if filtered else evidence_list


# # ─────────────────────────────────────────
# # 4. CONNECTION TEST
# # ─────────────────────────────────────────
# def test_llm_connection():
#     """Run this once to confirm Gemini API is working."""
#     print("🧪 Testing Gemini connection...")
#     try:
#         response = client.models.generate_content(
#             model=MODEL,
#             contents='Return this exact JSON and nothing else: {"status": "ok"}'
#         )
#         print("✅ Gemini is working. Response:", response.text)
#         return True
#     except Exception as e:
#         print(f"❌ Gemini FAILED: {type(e).__name__}: {e}")
#         print("👉 Check: Is API key valid? Quota exceeded? Network blocked?")
#         return False


# # ─────────────────────────────────────────
# # 5. MAIN PREDICTION FUNCTION
# # ─────────────────────────────────────────
# def predict_with_llm(claim, evidence_list):
#     """
#     Predict SUPPORTS / REFUTES / NOT ENOUGH INFO with real score distribution.
#     Falls back to keyword scoring if LLM is unavailable.
#     """

#     # ── No evidence at all ──
#     if not evidence_list:
#         print("⚠️ No evidence provided.")
#         return {
#             "label": "NOT ENOUGH INFO",
#             "scores": {"SUPPORTS": 5.0, "REFUTES": 5.0, "NOT ENOUGH INFO": 90.0}
#         }

#     # ── Filter irrelevant evidence ──
#     evidence_list = filter_relevant_evidence(claim, evidence_list)

#     # ── Build numbered evidence text ──
#     evidence_text = "\n".join([f"{i+1}. {e}" for i, e in enumerate(evidence_list[:5])])

#     prompt = f"""You are a fact-checking system. Your job is to verify whether the CLAIM is supported or refuted by the EVIDENCE.

# CLAIM: "{claim}"

# EVIDENCE:
# {evidence_text}

# INSTRUCTIONS:
# - Read each evidence sentence carefully.
# - Decide if the evidence SUPPORTS, REFUTES, or is unrelated to the claim.
# - Assign scores out of 100 that reflect the actual evidence. NEVER return equal scores like 33/33/33.
# - If evidence clearly supports the claim → SUPPORTS must be the highest score (60-95).
# - If evidence clearly refutes the claim → REFUTES must be the highest score (60-95).
# - If evidence is vague or unrelated → NOT ENOUGH INFO must be highest (50-85).
# - Always reflect uncertainty: rarely give 0 to any category.

# IMPORTANT: Return ONLY raw JSON. No explanation. No markdown. No code blocks.

# Example output:
# {{"label": "SUPPORTS", "scores": {{"SUPPORTS": 75, "REFUTES": 10, "NOT ENOUGH INFO": 15}}}}"""

#     last_error = None

#     for attempt in range(3):
#         try:
#             print(f"\n🔁 LLM Attempt {attempt + 1}/3...")

#             response = client.models.generate_content(
#                 model=MODEL,
#                 contents=prompt,
#                 config=types.GenerateContentConfig(
#                     temperature=0.1,
#                     max_output_tokens=200,
#                 )
#             )

#             # ── Check for empty response ──
#             if not response or not response.text:
#                 print("⚠️ Empty response from Gemini.")
#                 last_error = "Empty response"
#                 continue

#             raw = response.text.strip()
#             print(f"📨 Raw output: {raw}")

#             data = safe_parse_json(raw)
#             if data is None:
#                 print("⚠️ JSON parse failed.")
#                 last_error = "JSON parse failed"
#                 continue

#             label = str(data.get("label", "")).strip().upper()
#             scores = data.get("scores", {})

#             # Handle multiple possible key formats
#             s = float(scores.get("SUPPORTS",    scores.get("supports", 0)))
#             r = float(scores.get("REFUTES",     scores.get("refutes", 0)))
#             n = float(scores.get("NOT ENOUGH INFO",
#                       scores.get("NOT_ENOUGH_INFO",
#                       scores.get("not enough info", 0))))

#             total = s + r + n
#             if total <= 0:
#                 print("⚠️ All scores are 0, retrying...")
#                 last_error = "Zero total score"
#                 continue

#             # Normalize to exactly 100
#             s = round((s / total) * 100, 2)
#             r = round((r / total) * 100, 2)
#             n = round(100.0 - s - r, 2)

#             # Fix label if invalid or missing
#             valid_labels = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
#             if label not in valid_labels:
#                 score_map = {"SUPPORTS": s, "REFUTES": r, "NOT ENOUGH INFO": n}
#                 label = max(score_map, key=lambda k: score_map[k])

#             print(f"✅ Result → {label} | SUPPORTS={s}% | REFUTES={r}% | NEI={n}%")

#             return {
#                 "label": label,
#                 "scores": {"SUPPORTS": s, "REFUTES": r, "NOT ENOUGH INFO": n}
#             }

#         except Exception as e:
#             print(f"❌ Exception on attempt {attempt + 1}: {type(e).__name__}: {e}")
#             last_error = str(e)

#     # ── All LLM attempts failed → keyword fallback ──
#     print(f"\n💀 All LLM attempts failed. Last error: {last_error}")
#     print("🔎 Using keyword-based fallback scoring...")
#     return _keyword_fallback(claim, evidence_list)


# # ─────────────────────────────────────────
# # 6. QUICK TEST
# # ─────────────────────────────────────────
# if __name__ == "__main__":

#     # Step 1: Test connection first
#     if not test_llm_connection():
#         print("\n🚨 Fix your API key / network before continuing.")
#     else:
#         # Step 2: Test with sample claims
#         test_cases = [
#             {
#                 "claim": "Earth is a planet",
#                 "evidence": [
#                     "Earth is the third planet from the Sun.",
#                     "Earth orbits the Sun and is classified as a terrestrial planet.",
#                     "Earth has one natural satellite, the Moon.",
#                 ]
#             },
#             {
#                 "claim": "The moon is made of cheese",
#                 "evidence": [
#                     "The Moon is composed of rock, dust, and minerals.",
#                     "Lunar samples brought back by Apollo missions contain no organic material.",
#                     "The Moon's surface is covered in regolith, a type of rocky soil.",
#                 ]
#             },
#             {
#                 "claim": "Water boils at 100 degrees Celsius",
#                 "evidence": [
#                     "At sea level, water boils at 100°C (212°F).",
#                     "Boiling point decreases at higher altitudes due to lower air pressure.",
#                 ]
#             },
#         ]

#         for i, tc in enumerate(test_cases):
#             print(f"\n{'='*60}")
#             print(f"🧪 TEST {i+1}: {tc['claim']}")
#             print(f"{'='*60}")
#             result = predict_with_llm(tc["claim"], tc["evidence"])
#             print(f"\n📊 FINAL RESULT:")
#             print(f"   Label    : {result['label']}")
#             print(f"   SUPPORTS : {result['scores']['SUPPORTS']}%")
#             print(f"   REFUTES  : {result['scores']['REFUTES']}%")
#             print(f"   NEI      : {result['scores']['NOT ENOUGH INFO']}%")


##-------------------------------------------------

from google import genai
from google.genai import types
import json

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
client = genai.Client(api_key="AIzaSyDhSMYIavrWceskOe6zoEk5_cRH0ogr7kE")
MODEL = "gemini-1.5-flash-latest"


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def safe_parse_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except:
        pass
    return None


def _keyword_fallback(claim, evidence_list):
    """Used ONLY when Gemini API is completely unreachable."""
    claim_words = set(claim.lower().split())
    support_hits, refute_hits = 0, 0

    refute_words  = {"not","never","false","incorrect","wrong","no","cannot","isn't","aren't","wasn't","doesn't","don't"}
    support_words = {"is","was","are","confirmed","true","correct","yes","indeed","does","has","have","will"}

    for ev in evidence_list[:5]:
        ev_words = set(ev.lower().split())
        overlap  = len(claim_words & ev_words) / max(len(claim_words), 1)
        if overlap > 0.25:
            if ev_words & refute_words:
                refute_hits += 1
            elif ev_words & support_words:
                support_hits += 1

    total = support_hits + refute_hits
    if total == 0:
        return {"label": "NOT ENOUGH INFO",
                "scores": {"SUPPORTS": 10.0, "REFUTES": 10.0, "NOT ENOUGH INFO": 80.0},
                "summary": "No relevant evidence found for this claim."}

    s = round((support_hits / total) * 70, 2)
    r = round((refute_hits  / total) * 70, 2)
    n = round(100.0 - s - r, 2)
    score_map = {"SUPPORTS": s, "REFUTES": r, "NOT ENOUGH INFO": n}
    label = max(score_map, key=lambda k: score_map[k])
    return {"label": label, "scores": score_map,
            "summary": f"Keyword analysis: {support_hits} supporting, {refute_hits} refuting evidence sentences."}


# ─────────────────────────────────────────
# MAIN FUNCTION  (called from app.py)
# ─────────────────────────────────────────
def predict_with_llm(claim, evidence_list):
    """
    Returns:
      {
        "label":   "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
        "scores":  {"SUPPORTS": float, "REFUTES": float, "NOT ENOUGH INFO": float},
        "summary": str   ← shown in the Summary section of app.py
      }
    """

    # ── No evidence ──
    if not evidence_list:
        return {
            "label": "NOT ENOUGH INFO",
            "scores": {"SUPPORTS": 5.0, "REFUTES": 5.0, "NOT ENOUGH INFO": 90.0},
            "summary": "No evidence was retrieved for this claim."
        }

    evidence_text = "\n".join([f"{i+1}. {e}" for i, e in enumerate(evidence_list[:5])])

    # ── TWO separate prompts for two jobs ──

    # --- Prompt A: Classification + Scores ---
    score_prompt = f"""You are a fact-checking AI. Analyze the claim against the evidence.

CLAIM: "{claim}"

EVIDENCE:
{evidence_text}

TASK: Return a JSON object with your verdict and confidence scores (must sum to 100).

SCORING RULES:
- Evidence clearly confirms claim  → SUPPORTS: 65-92, REFUTES: 3-15, NOT ENOUGH INFO: 5-20
- Evidence clearly contradicts claim → REFUTES: 65-92, SUPPORTS: 3-15, NOT ENOUGH INFO: 5-20
- Evidence is vague or off-topic   → NOT ENOUGH INFO: 55-80, others share remaining
- NEVER return equal scores like 33/33/33 or 10/10/80

Return ONLY this JSON, no markdown, no explanation:
{{"label": "SUPPORTS", "scores": {{"SUPPORTS": 78, "REFUTES": 8, "NOT ENOUGH INFO": 14}}}}"""

    # --- Prompt B: Human-readable summary ---
    summary_prompt = f"""You are a fact-checking assistant. Write a 2-3 sentence summary explaining the verdict for this claim.

CLAIM: "{claim}"
EVIDENCE:
{evidence_text}

Be specific — mention what the evidence says. Do not use bullet points. Plain sentences only."""

    last_error = None
    result_scores = None
    result_summary = None

    # ── Get scores ──
    for attempt in range(3):
        try:
            print(f"\n🔁 Score attempt {attempt+1}/3...")
            response = client.models.generate_content(
                model=MODEL,
                contents=score_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=150,
                )
            )

            if not response or not response.text:
                last_error = "Empty response"
                continue

            raw = response.text.strip()
            print(f"📨 Raw: {raw}")

            data = safe_parse_json(raw)
            if data is None:
                last_error = "JSON parse failed"
                continue

            label  = str(data.get("label", "")).strip().upper()
            scores = data.get("scores", {})

            s = float(scores.get("SUPPORTS",         scores.get("supports", 0)))
            r = float(scores.get("REFUTES",          scores.get("refutes", 0)))
            n = float(scores.get("NOT ENOUGH INFO",
                      scores.get("NOT_ENOUGH_INFO",
                      scores.get("not enough info",  0))))

            total = s + r + n
            if total <= 0:
                last_error = "Zero scores"
                continue

            s = round((s / total) * 100, 2)
            r = round((r / total) * 100, 2)
            n = round(100.0 - s - r, 2)

            valid = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
            if label not in valid:
                label = max({"SUPPORTS": s, "REFUTES": r, "NOT ENOUGH INFO": n},
                            key=lambda k: {"SUPPORTS": s, "REFUTES": r, "NOT ENOUGH INFO": n}[k])

            result_scores = {"label": label,
                             "scores": {"SUPPORTS": s, "REFUTES": r, "NOT ENOUGH INFO": n}}
            print(f"✅ {label} | S={s}% R={r}% N={n}%")
            break

        except Exception as e:
            print(f"❌ Attempt {attempt+1} error: {e}")
            last_error = str(e)

    # ── Get summary ──
    try:
        sum_response = client.models.generate_content(
            model=MODEL,
            contents=summary_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200,
            )
        )
        result_summary = sum_response.text.strip() if sum_response and sum_response.text else None
    except Exception as e:
        print(f"⚠️ Summary generation failed: {e}")
        result_summary = None

    # ── Fallback if scores failed ──
    if result_scores is None:
        print(f"💀 All score attempts failed ({last_error}). Using keyword fallback.")
        fallback = _keyword_fallback(claim, evidence_list)
        fallback["summary"] = result_summary or fallback["summary"]
        return fallback

    # ── Build default summary if LLM summary failed ──
    if result_summary is None:
        label = result_scores["label"]
        s = result_scores["scores"]
        result_summary = (
            f"The claim '{claim}' was analyzed against {len(evidence_list)} evidence sentences. "
            f"Verdict: {label} "
            f"(Supports: {s['SUPPORTS']}%, Refutes: {s['REFUTES']}%, Not Enough Info: {s['NOT ENOUGH INFO']}%)."
        )

    return {
        "label":   result_scores["label"],
        "scores":  result_scores["scores"],
        "summary": result_summary
    }