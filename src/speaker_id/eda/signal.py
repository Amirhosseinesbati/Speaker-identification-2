"""Streaming waveform statistics; energy heuristics are not speech/VAD or SNR."""

from __future__ import annotations

import numpy as np

try:
    import webrtcvad
except ImportError:  # Optional EDA extra; never required by inference audio I/O.
    webrtcvad = None

DB_FLOOR = -160.0
CLIP_LEVEL = 1.0 - 1.0 / 32768.0
FRAME_SECONDS = 0.020


def amplitude_db(value):
    return np.maximum(20 * np.log10(np.maximum(value, 1e-8)), DB_FLOOR)


class SignalAccumulator:
    """Keep block memory plus 50 short-frame energies per audio second."""

    def __init__(self, sample_rate: int, channels: int):
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_size = max(1, round(sample_rate * FRAME_SECONDS))
        self.frames = 0
        self.total = np.zeros(channels, dtype=np.float64)
        self.squares = np.zeros(channels, dtype=np.float64)
        self.peak = np.zeros(channels, dtype=np.float64)
        self.zeros = np.zeros(channels, dtype=np.int64)
        self.clips = np.zeros(channels, dtype=np.int64)
        self.nonfinite = 0
        self.cross = 0.0
        self.identical = channels > 1
        self.mono_sum = self.mono_square = self.mono_peak = 0.0
        self.mono_zeros = self.mono_crossings = 0
        self.previous_mono_negative = None
        self.channel_crossings = np.zeros(channels, dtype=np.int64)
        self.previous_channel_negative = None
        self.pending = np.empty((0, channels), dtype=np.float32)
        self.energies = []
        self.vad_frame_size = round(sample_rate * .030)
        self.vad_available = webrtcvad is not None and sample_rate in {8000, 16000, 32000, 48000}
        self.vad_models = {mode: [webrtcvad.Vad(mode) for _ in range(channels)] for mode in (1, 3)} if self.vad_available else {}
        self.vad_positive = {mode: np.zeros(channels, dtype=np.int64) for mode in (1, 3)}
        self.vad_frames = 0
        self.vad_pending = np.empty((0, channels), dtype=np.float32)

    def update(self, samples: np.ndarray) -> None:
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise ValueError("Expected (frames, channels) audio blocks")
        if not len(samples):
            return
        self.nonfinite += int(np.count_nonzero(~np.isfinite(samples)))
        # Non-finite streams remain invalid; replacements only keep diagnostics serializable.
        samples = np.nan_to_num(samples, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        x = samples.astype(np.float64)
        self.frames += len(x)
        self.total += x.sum(axis=0)
        self.squares += np.einsum("ij,ij->j", x, x)
        absolute = np.abs(x)
        self.peak = np.maximum(self.peak, absolute.max(axis=0))
        self.zeros += (x == 0).sum(axis=0)
        self.clips += (absolute >= CLIP_LEVEL).sum(axis=0)
        if self.channels > 1:
            self.identical = self.identical and bool(np.all(samples == samples[:, :1]))
            self.cross += float(np.dot(x[:, 0], x[:, 1]))
        negative = x < 0
        self.channel_crossings += np.count_nonzero(negative[1:] != negative[:-1], axis=0)
        if self.previous_channel_negative is not None:
            self.channel_crossings += negative[0] != self.previous_channel_negative
        self.previous_channel_negative = negative[-1].copy()
        mono = x.mean(axis=1)
        self.mono_sum += float(mono.sum())
        self.mono_square += float(np.dot(mono, mono))
        self.mono_peak = max(self.mono_peak, float(np.max(np.abs(mono))))
        self.mono_zeros += int(np.count_nonzero(mono == 0))
        mono_negative = mono < 0
        self.mono_crossings += int(np.count_nonzero(mono_negative[1:] != mono_negative[:-1]))
        if self.previous_mono_negative is not None:
            self.mono_crossings += int(mono_negative[0] != self.previous_mono_negative)
        self.previous_mono_negative = bool(mono_negative[-1])
        framed = np.concatenate((self.pending, samples), axis=0)
        count = len(framed) // self.frame_size
        if count:
            complete = framed[:count * self.frame_size].reshape(count, self.frame_size, self.channels)
            self.energies.append(np.einsum("ijk,ijk->ik", complete, complete, dtype=np.float64) / self.frame_size)
        self.pending = framed[count * self.frame_size:].copy()
        if self.vad_available:
            vad_audio = np.concatenate((self.vad_pending, samples), axis=0)
            vad_count = len(vad_audio) // self.vad_frame_size
            if vad_count:
                vad_pcm = np.clip(np.rint(vad_audio[:vad_count * self.vad_frame_size] * 32768), -32768, 32767).astype("<i2")
                vad_pcm = vad_pcm.reshape(vad_count, self.vad_frame_size, self.channels)
                for channel in range(self.channels):
                    for frame in vad_pcm[:, :, channel]:
                        payload = frame.tobytes()
                        for mode in (1, 3):
                            self.vad_positive[mode][channel] += self.vad_models[mode][channel].is_speech(payload, self.sample_rate)
                self.vad_frames += vad_count
            self.vad_pending = vad_audio[vad_count * self.vad_frame_size:].copy()

    def finalize(self) -> dict:
        n = max(1, self.frames)
        rms = np.sqrt(self.squares / n)
        dc = self.total / n
        mono_rms = np.sqrt(self.mono_square / n)
        channel_index = int(np.argmax(rms))
        correlation = None
        if self.channels > 1 and self.frames > 1:
            variance = np.maximum(self.squares / n - dc ** 2, 0)
            denominator = np.sqrt(variance[0] * variance[1])
            if denominator > 0:
                correlation = float(np.clip((self.cross / n - dc[0] * dc[1]) / denominator, -1, 1))
        energy_arrays = list(self.energies)
        if len(self.pending):
            energy_arrays.append(np.mean(self.pending.astype(np.float64) ** 2, axis=0, keepdims=True))
        energy = np.concatenate(energy_arrays, axis=0)[:, channel_index] if energy_arrays else np.array([])
        energy_db = amplitude_db(np.sqrt(energy))
        percentiles = np.percentile(energy_db, [1, 10, 50, 90, 99]) if len(energy_db) else [None] * 5
        result = {
            "decoded_frames": self.frames,
            "duration_seconds": self.frames / self.sample_rate,
            "nonfinite_samples": self.nonfinite,
            "has_nonzero_signal": bool(self.peak.max() > 0),
            "channel_identical": self.identical if self.channels > 1 else None,
            "channel_correlation": correlation,
            "channel_rms_imbalance_db": float(amplitude_db(rms.max()) - amplitude_db(rms.min())),
            "downmix_attenuation_db": float(amplitude_db(mono_rms) - amplitude_db(rms.max())),
            "channel_peak_dbfs": amplitude_db(self.peak).tolist(),
            "channel_rms_dbfs": amplitude_db(rms).tolist(),
            "channel_dc": dc.tolist(),
            "channel_clip_fraction": (self.clips / n).tolist(),
            "channel_zero_fraction": (self.zeros / n).tolist(),
            "max_channel_clip_fraction": float(self.clips.max() / n),
            "max_channel_rms_dbfs": float(amplitude_db(rms.max())),
            "max_channel_abs_dc": float(np.abs(dc).max()),
            "mono_peak_dbfs": float(amplitude_db(self.mono_peak)),
            "mono_rms_dbfs": float(amplitude_db(mono_rms)),
            "mono_dc": self.mono_sum / n,
            "mono_zero_fraction": self.mono_zeros / n,
            "mono_zcr": self.mono_crossings / max(1, self.frames - 1),
            "analysis_channel_index": channel_index,
            "analysis_zcr": float(self.channel_crossings[channel_index] / max(1, self.frames - 1)),
            "energy_frame_count": len(energy_db),
        }
        for percentile, value in zip([1, 10, 50, 90, 99], percentiles):
            result[f"frame_energy_p{percentile}_dbfs"] = None if value is None else float(value)
        for threshold in [60, 50, 40]:
            result[f"below_minus{threshold}_dbfs_fraction"] = float(np.mean(energy_db < -threshold)) if len(energy_db) else None
        result["relative_low_energy_fraction"] = float(np.mean(energy_db < percentiles[3] - 30)) if len(energy_db) else None
        result["energy_above_minus50_dbfs_seconds"] = float(np.count_nonzero(energy_db >= -50) * self.frame_size / self.sample_rate)
        # A partial final frame contributes its actual length, not another full 20 ms.
        if len(self.pending) and len(energy_db) and energy_db[-1] >= -50:
            result["energy_above_minus50_dbfs_seconds"] -= (self.frame_size - len(self.pending)) / self.sample_rate
        result["vad_status"] = "ok" if self.vad_available else "unavailable_or_unsupported_sample_rate"
        result["vad_analyzed_seconds"] = self.vad_frames * .030 if self.vad_available else 0.0
        result["vad_unprocessed_tail_seconds"] = len(self.vad_pending) / self.sample_rate if self.vad_available else self.frames / self.sample_rate
        for mode in (1, 3):
            counts = self.vad_positive[mode]
            result[f"channel_vad_mode{mode}_speech_seconds"] = (counts * .030).tolist() if self.vad_available else None
            result[f"vad_mode{mode}_speech_seconds"] = float(counts[channel_index] * .030) if self.vad_available else None
            result[f"vad_mode{mode}_speech_fraction"] = float(counts[channel_index] / self.vad_frames) if self.vad_frames else None
        return result
