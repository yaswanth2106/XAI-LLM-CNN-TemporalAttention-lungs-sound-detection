import json
import cohere
import os


co = cohere.Client(api_key="agNom6U3QAMfQKZ1iB1l6xkXpRhpKmtdJWscBZZa")


def generate_clinical_explanation(explanation_json):
    system_msg = (
        "You are an expert clinical AI assistant. Convert model XAI outputs "
        "into a concise, clinician-friendly explanation. Never make a definitive diagnosis. "
        "Always add a disclaimer asking for clinical confirmation."
    )

    user_prompt = f"""
Input (JSON):
{json.dumps(explanation_json, indent=2)}

Produce:
1) Summary: predicted class & key spectrogram findings.
2) Findings: 3 bullets describing time regions.
3) Suggested next steps: 2 bullets.
4) Disclaimer sentence.
"""

    response = co.chat(
        model="command-xlarge-nightly",
        message=user_prompt,
        preamble=system_msg,
        temperature=0.3,
        max_tokens=400
    )

    return response.text
