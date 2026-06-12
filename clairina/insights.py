# clairina/insights.py
import json, os
from datetime import datetime

def build_model_payload(summary: dict, channels: list, baseline: dict) -> str:
    payload = {"summary": summary, "channels": channels, "baseline": baseline}
    return json.dumps(payload, indent=2)

def format_prompt(payload_json: str) -> str:
    return f"""
You are a marketing analyst. Given the JSON below, produce:
1) A 2-sentence executive summary of highest-impact channels.
2) Three tactical recommendations (one-line each).
3) One "what-if" sentence for moving 5% budget from worst to best channel.
4) One-line about confidence (mention CI width).

Return valid JSON with keys: summary, recommendations, what_if, confidence.

Input:
{payload_json}
"""

def log_llm_call(payload, response, path="data/llm_calls.jsonl"):
    record = {"ts": datetime.utcnow().isoformat(), "payload": payload, "response": response}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(record) + "\n")

def mock_generate(payload_json: str):
    parsed = json.loads(payload_json)
    chs = parsed.get("channels", [])[:]
    return {
        "summary": "Google and Meta appear to drive the most incremental revenue; TV and organic have lower marginal impact.",
        "recommendations": [
            "Shift a small percentage of TV spend to Google and monitor lift.",
            "Run creative experiments on TikTok to improve conversion.",
            "Use email for retention campaigns and reduce paid retargeting slightly."
        ],
        "what_if": "Moving ~5% from TV to Google could increase short-term revenue by ~1-3% (approx).",
        "confidence": "Model fit is moderate; check CI widths for each channel before acting."
    }

def generate_insights(summary: dict, channels: list, baseline: dict, provider="mock", **kwargs):
    payload = build_model_payload(summary, channels, baseline)
    prompt = format_prompt(payload)
    if provider == "mock":
        resp = mock_generate(payload)
        log_llm_call(payload, resp)
        return resp
    if provider == "openai":
        try:
            import openai
        except Exception as e:
            raise RuntimeError("openai package not installed")
        openai.api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))
        reply = openai.ChatCompletion.create(
            model=kwargs.get("model","gpt-4o-mini"),
            messages=[{"role":"system","content":"You are a helpful marketing analyst."},
                      {"role":"user","content":prompt}],
            temperature=0.2, max_tokens=400
        )
        text = reply.choices[0].message.content
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"raw": text}
        log_llm_call(payload, parsed)
        return parsed
    raise ValueError("unknown provider")
