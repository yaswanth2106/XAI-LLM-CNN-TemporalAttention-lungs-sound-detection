

import numpy as np
import librosa


def assess_audio_quality(audio_data, sr):
    
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    duration_sec = len(audio_data) / sr
    frame_length = int(0.025 * sr)  
    hop_length = int(0.010 * sr)    

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

    clipping_ratio = float((np.abs(audio_data) > 0.99).mean())
    has_clipping = clipping_ratio > 0.001   

    silence_threshold = 0.01
    silence_ratio = (
        float((rms < silence_threshold).mean()) if len(rms) > 0 else 1.0
    )

    if duration_sec < 2.0:
        duration_quality = "too_short"
    elif duration_sec < 3.0:
        duration_quality = "short"
    elif duration_sec <= 10.0:
        duration_quality = "adequate"
    else:
        duration_quality = "long"

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
