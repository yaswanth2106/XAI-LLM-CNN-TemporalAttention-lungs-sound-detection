"""
Rule-Based Clinical Explanation Engine

Generates deterministic, auditable clinical explanations from
enriched XAI metrics. Replaces the LLM entirely.

Properties:
  - Zero cloud dependencies
  - Sub-millisecond latency
  - 100% deterministic (same input → same output, always)
  - Fully auditable and unit-testable
  - Edge-device ready
"""


# ── Condition Knowledge Base ──────────────────────────────────────────────
# Curated by domain context — every string is intentional and traceable.

CONDITION_CONTEXT = {
    "Asthma": {
        "full_name": "Asthma",
        "icd10": "J45",
        "description": (
            "a chronic inflammatory airway disease characterized "
            "by variable airflow obstruction"
        ),
        "typical_findings": (
            "episodic wheezing with variable airflow obstruction"
        ),
        "sound_characteristics": (
            "polyphonic wheezing, predominantly during expiration"
        ),
        "next_steps": [
            "Consider peak flow monitoring and bronchodilator "
            "response testing",
            "Assess for environmental or allergic triggers",
            "Evaluate with spirometry if not recently performed"
        ]
    },
    "COPD": {
        "full_name": "Chronic Obstructive Pulmonary Disease",
        "icd10": "J44",
        "description": (
            "a progressive lung disease characterized by "
            "persistent airflow limitation"
        ),
        "typical_findings": (
            "prolonged expiratory phase and wheezing patterns"
        ),
        "sound_characteristics": (
            "diminished breath sounds with expiratory wheezing "
            "and possible rhonchi"
        ),
        "next_steps": [
            "Consider spirometry for FEV1/FVC ratio assessment",
            "Review patient history for smoking or occupational exposure",
            "Assess oxygen saturation and consider arterial blood "
            "gas analysis"
        ]
    },
    "Healthy": {
        "full_name": "Healthy / Normal Breath Sounds",
        "icd10": None,
        "description": (
            "normal vesicular breath sounds without "
            "adventitious sounds"
        ),
        "typical_findings": (
            "clear vesicular sounds throughout the respiratory cycle"
        ),
        "sound_characteristics": (
            "soft, low-pitched sounds throughout inspiration "
            "with no adventitious sounds"
        ),
        "next_steps": [
            "No immediate action required based on lung sound analysis",
            "Continue routine monitoring as appropriate for "
            "patient context"
        ]
    },
    "ILD": {
        "full_name": "Interstitial Lung Disease",
        "icd10": "J84",
        "description": (
            "a group of disorders causing progressive scarring "
            "of lung tissue"
        ),
        "typical_findings": (
            "fine inspiratory crackles, typically bibasilar"
        ),
        "sound_characteristics": (
            "fine, velcro-like crackles predominantly during "
            "late inspiration"
        ),
        "next_steps": [
            "Consider high-resolution CT (HRCT) of the chest "
            "for detailed assessment",
            "Evaluate pulmonary function tests including DLCO",
            "Review for occupational, environmental, or "
            "autoimmune etiologies"
        ]
    },
    "Infection": {
        "full_name": "Respiratory Infection",
        "icd10": "J06/J18",
        "description": (
            "an acute or chronic infection of the respiratory tract"
        ),
        "typical_findings": (
            "coarse crackles, bronchial breath sounds, or "
            "consolidation patterns"
        ),
        "sound_characteristics": (
            "coarse crackles, possibly with bronchial breathing "
            "and egophony over affected areas"
        ),
        "next_steps": [
            "Consider chest X-ray to assess for consolidation "
            "or infiltrates",
            "Evaluate inflammatory markers (CBC, CRP, procalcitonin)",
            "Assess need for sputum culture and antimicrobial therapy"
        ]
    }
}


# ── Confidence Descriptions ──────────────────────────────────────────────

CONFIDENCE_DESCRIPTIONS = {
    "high": (
        "The model shows **high confidence** in this prediction"
    ),
    "moderate": (
        "The model shows **moderate confidence** in this prediction"
    ),
    "low": (
        "The model shows **low confidence** in this prediction "
        "— interpret with caution"
    ),
    "very_low": (
        "The model shows **very low confidence** — this should be "
        "treated as a preliminary screening suggestion only"
    )
}


# ── Helper Functions ─────────────────────────────────────────────────────

def describe_severity(score):
    """Map Grad-CAM severity score to a clinical descriptor."""
    if score >= 0.85:
        return "high-intensity"
    elif score >= 0.60:
        return "moderate-intensity"
    elif score >= 0.35:
        return "mild-intensity"
    else:
        return "low-intensity"


def describe_agreement(score):
    """Describe the attention–CAM agreement level."""
    if score >= 0.8:
        return "strong"
    elif score >= 0.5:
        return "moderate"
    else:
        return "weak"


# ── Core Explanation Generator ────────────────────────────────────────────

def generate_clinical_explanation(explanation_json):
    """Generate a deterministic clinical explanation from enriched XAI JSON.

    This function replaces the LLM entirely. Every output is
    reproducible, auditable, and runs in <1ms with zero cloud
    dependencies.

    Args:
        explanation_json: dict from xai.build_explanation()

    Returns:
        str: formatted clinical report in markdown
    """
    cls = explanation_json["predicted_class"]
    prob = explanation_json["probability"]
    tier = explanation_json["confidence_tier"]
    regions = explanation_json["top_time_regions_sec"]
    cam_stats = explanation_json["cam_statistics"]
    attn_agreement = explanation_json["attention_cam_agreement"]
    audio_q = explanation_json["audio_quality"]
    ensemble = explanation_json.get("ensemble")
    differential = explanation_json.get("differential")

    ctx = CONDITION_CONTEXT.get(cls, CONDITION_CONTEXT["Healthy"])
    sections = []

    # ── 1. Audio Quality Warning (if needed) ─────────────────────────
    if audio_q.get("overall_quality") == "poor":
        quality_warnings = []
        for issue in audio_q.get("issues", []):
            if issue == "low_snr":
                quality_warnings.append(
                    f"Low signal-to-noise ratio "
                    f"({audio_q['snr_db']:.0f} dB) — background "
                    "noise may affect accuracy"
                )
            elif issue == "clipping":
                quality_warnings.append(
                    "Audio clipping detected — signal distortion "
                    "may affect feature extraction"
                )
            elif issue == "excessive_silence":
                quality_warnings.append(
                    f"Recording is "
                    f"{audio_q['silence_ratio']:.0%} silence — "
                    "insufficient respiratory signal"
                )
            elif issue == "insufficient_duration":
                quality_warnings.append(
                    f"Recording is only "
                    f"{audio_q['duration_sec']:.1f}s — longer "
                    "samples (≥3s) recommended for reliable analysis"
                )
        if quality_warnings:
            warning_text = "⚠️ **Audio Quality Concerns:**\n"
            warning_text += "\n".join(
                f"- {w}" for w in quality_warnings
            )
            sections.append(warning_text)

    elif audio_q.get("overall_quality") == "acceptable":
        issues_text = ", ".join(audio_q.get("issues", []))
        sections.append(
            f"ℹ️ *Audio quality note: {issues_text} detected. "
            "Results may be marginally affected.*"
        )

    # ── 2. Summary ───────────────────────────────────────────────────
    confidence_desc = CONFIDENCE_DESCRIPTIONS.get(
        tier, CONFIDENCE_DESCRIPTIONS["low"]
    )
    summary = "### Summary\n\n"
    summary += f"**Predicted condition:** {ctx['full_name']}"
    if ctx.get("icd10"):
        summary += f" (ICD-10: {ctx['icd10']})"
    summary += "\n\n"
    summary += f"{confidence_desc} ({prob:.1%} probability)."

    # Entropy context
    entropy = explanation_json.get("prediction_entropy", 0)
    if entropy > 0.7:
        summary += (
            f" The prediction entropy is elevated ({entropy:.2f}), "
            "indicating the model is uncertain across multiple "
            "classes."
        )
    elif entropy < 0.2:
        summary += (
            f" The prediction entropy is very low ({entropy:.2f}), "
            "indicating a decisive classification."
        )

    sections.append(summary)

    # ── 3. Ensemble Consensus (if available) ─────────────────────────
    if ensemble:
        ensemble_section = "### Ensemble Consensus\n\n"
        agreement = ensemble["agreement_count"]
        total = ensemble["n_folds"]

        if ensemble.get("consensus"):
            ensemble_section += (
                f"✅ **Strong consensus:** {agreement} out of "
                f"{total} models agree on **{cls}** "
                f"(ensemble std: ±{ensemble['ensemble_std']:.3f})."
            )
        else:
            counts = ensemble.get("prediction_counts", {})
            disagreements = [
                f"{name} ({count}/{total} folds)"
                for name, count in sorted(
                    counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
            ensemble_section += (
                f"⚠️ **Model disagreement detected:** Only "
                f"{agreement}/{total} folds agree.\n\n"
                f"Fold predictions: "
                f"{', '.join(disagreements)}.\n\n"
                "Inter-model disagreement suggests this case "
                "may be atypical or borderline."
            )
        sections.append(ensemble_section)

    # ── 4. Key Findings ──────────────────────────────────────────────
    findings_section = "### Key Findings\n\n"

    if regions:
        for i, r in enumerate(regions, 1):
            sev = describe_severity(r["severity_score"])
            findings_section += (
                f"- **Region {i}** "
                f"({r['start']:.2f}s – {r['end']:.2f}s): "
                f"{sev} activation "
                f"(score: {r['severity_score']:.2f}), "
                f"consistent with {ctx['typical_findings']}.\n"
            )
    else:
        findings_section += (
            "- No significant activation regions detected "
            "above threshold.\n"
        )

    # CAM pattern description
    spread = cam_stats.get("spread_label", "unknown")
    coverage = cam_stats.get("coverage_ratio", 0)
    n_regions = cam_stats.get("n_active_regions", 0)

    findings_section += "\n**Activation pattern:** "
    if spread == "focal":
        findings_section += (
            f"Focal activation in a single region "
            f"({coverage:.0%} of signal)."
        )
    elif spread == "clustered":
        findings_section += (
            f"{n_regions} activated regions clustered together "
            f"({coverage:.0%} of signal), suggesting a "
            "localized abnormality."
        )
    elif spread == "spread":
        findings_section += (
            f"{n_regions} activated regions spread across the "
            f"recording ({coverage:.0%} of signal), suggesting "
            "a diffuse or recurring abnormality."
        )
    elif spread == "none":
        findings_section += (
            "No significant activation regions detected."
        )
    else:
        findings_section += (
            f"{n_regions} region(s) detected, covering "
            f"{coverage:.0%} of the signal."
        )

    # Attention–CAM agreement
    agreement_label = describe_agreement(attn_agreement)
    findings_section += (
        f"\n\n**Model consistency:** {agreement_label} agreement "
        f"between temporal attention and Grad-CAM "
        f"(cosine similarity: {attn_agreement:.2f})."
    )
    if attn_agreement >= 0.7:
        findings_section += (
            " Both model mechanisms focus on the same regions, "
            "increasing confidence in the highlighted areas."
        )
    elif attn_agreement < 0.4:
        findings_section += (
            " The model's attention and gradient-based "
            "explanations diverge — interpret highlighted "
            "regions with additional caution."
        )

    sections.append(findings_section)

    # ── 5. Differential Diagnosis (if applicable) ────────────────────
    top2_gap = explanation_json.get("top2_gap", 1.0)
    second_class = explanation_json.get("second_class", "")
    second_prob = explanation_json.get("second_probability", 0)

    if top2_gap < 0.3 and second_class:
        second_ctx = CONDITION_CONTEXT.get(second_class, {})
        diff_section = "### Differential Diagnosis\n\n"
        diff_section += (
            f"⚠️ The model also considers "
            f"**{second_ctx.get('full_name', second_class)}** "
            f"({second_prob:.1%} probability). The gap between "
            f"the top two predictions is narrow "
            f"({top2_gap:.1%}).\n\n"
        )

        if (differential
                and differential.get("distinguishing_regions")):
            diff_section += "**Distinguishing regions:**\n"
            for dr in differential["distinguishing_regions"]:
                diff_section += (
                    f"- {dr['start']:.2f}s – {dr['end']:.2f}s: "
                    f"favors {cls} over {second_class} "
                    f"(differential score: "
                    f"{dr['diff_score']:.2f})\n"
                )

        diff_section += (
            f"\nClinical correlation is strongly recommended "
            f"to distinguish between {ctx['full_name']} and "
            f"{second_ctx.get('full_name', second_class)}."
        )
        sections.append(diff_section)

    # ── 6. Suggested Next Steps ──────────────────────────────────────
    steps_section = "### Suggested Next Steps\n\n"
    for step in ctx["next_steps"]:
        steps_section += f"- {step}\n"

    # Context-dependent extra steps
    if tier in ("low", "very_low"):
        steps_section += (
            "- **Repeat recording** with improved placement "
            "and reduced background noise\n"
        )
    if top2_gap < 0.3:
        steps_section += (
            "- **Additional diagnostic testing** to "
            "differentiate between competing diagnoses\n"
        )

    sections.append(steps_section)

    # ── 7. Probability Distribution Table ────────────────────────────
    dist = explanation_json.get("full_distribution", {})
    if dist:
        dist_section = "### Full Probability Distribution\n\n"
        dist_section += "| Condition | Probability |\n"
        dist_section += "|:--|--:|\n"
        for name, p in sorted(
            dist.items(), key=lambda x: x[1], reverse=True
        ):
            marker = " ◄" if name == cls else ""
            dist_section += f"| {name} | {p:.1%}{marker} |\n"
        sections.append(dist_section)

    # ── 8. Disclaimer (always identical — auditable) ─────────────────
    disclaimer = (
        "---\n\n"
        "*⚠️ **Disclaimer:** This is an AI-assisted screening "
        "tool and does not constitute a medical diagnosis. The "
        "analysis is based on acoustic features extracted from "
        "the provided audio sample and may not capture all "
        "clinically relevant information. All findings must be "
        "reviewed and confirmed by a qualified healthcare "
        "professional. Do not make treatment decisions based "
        "solely on this output.*"
    )
    sections.append(disclaimer)

    return "\n\n".join(sections)
