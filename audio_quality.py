"""
Audio Quality Assessment Module

Evaluates recording quality BEFORE inference so the rule engine
can qualify its own output. Runs entirely offline with zero
external dependencies beyond librosa/numpy.
"""

import numpy as np
import librosa


def assess_audio_quality(audio_data, sr):
    """Assess audio quality metrics for reliability estimation.

    These metrics let the rule engine decide whether to trust
    the model's prediction or add quality warnings.

    Args:
        audio_data: numpy array of audio samples (mono or stereo)
        sr: sample rate in Hz

    Returns:
        dict with SNR, clipping, silence ratio, duration, and
        overall quality rating
    """
    # Handle stereo → mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duration_sec = len(audio_data) / sr

    # ── SNR Estimation ──
    # Approximate: top 10% RMS as signal, bottom 10% as noise floor
    frame_length = int(0.025 * sr)   # 25ms frames
    hop_length = int(0.010 * sr)     # 10ms hop

    rms = librosa.feature.rms(
        y=audio_data, frame_length=frame_length, hop_length=hop_length
    )[0]

    if len(rms) > 0:
        sorted_rms = np.sort(rms)
        n_low = max(1, len(sorted_rms) // 10)
        n_high = max(1, len(sorted_rms) // 10)
        noise_floor = sorted_rms[:n_low].mean()
        signal_level = sorted_rms[-n_high:].mean()
        snr_db = 20 * np.log10(
            (signal_level + 1e-10) / (noise_floor + 1e-10)
        )
    else:
        snr_db = 0.0

    # ── Clipping Detection ──
    clipping_ratio = float((np.abs(audio_data) > 0.99).mean())
    has_clipping = clipping_ratio > 0.001   # more than 0.1% clipped

    # ── Silence Ratio ──
    # Frames below ~-40dB relative threshold
    silence_threshold = 0.01
    silence_ratio = (
        float((rms < silence_threshold).mean()) if len(rms) > 0 else 1.0
    )

    # ── Duration Adequacy ──
    if duration_sec < 2.0:
        duration_quality = "too_short"
    elif duration_sec < 3.0:
        duration_quality = "short"
    elif duration_sec <= 10.0:
        duration_quality = "adequate"
    else:
        duration_quality = "long"

    # ── Overall Quality Rating ──
    issues = []
    if snr_db < 10:
        issues.append("low_snr")
    if has_clipping:
        issues.append("clipping")
    if silence_ratio > 0.5:
        issues.append("excessive_silence")
    if duration_quality == "too_short":
        issues.append("insufficient_duration")

    if len(issues) == 0:
        overall = "good"
    elif len(issues) == 1:
        overall = "acceptable"
    else:
        overall = "poor"

    return {
        "snr_db": round(float(snr_db), 1),
        "clipping": has_clipping,
        "clipping_ratio": round(clipping_ratio, 5),
        "silence_ratio": round(silence_ratio, 3),
        "duration_sec": round(duration_sec, 2),
        "duration_quality": duration_quality,
        "issues": issues,
        "overall_quality": overall
    }
