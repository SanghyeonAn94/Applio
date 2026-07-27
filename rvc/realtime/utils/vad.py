import webrtcvad
import numpy as np


class VADProcessor:
    def __init__(self, sensitivity_mode=3, sample_rate=16000, frame_duration_ms=30):
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("VAD sample rate must be 8000, 16000, 32000, or 48000 Hz")
        if frame_duration_ms not in [10, 20, 30]:
            raise ValueError("VAD frame duration must be 10, 20, or 30 ms")

        self.vad = webrtcvad.Vad(sensitivity_mode)
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))

    def is_speech(self, audio_chunk_float32):
        if audio_chunk_float32.ndim > 1 and audio_chunk_float32.shape[1] == 1:
            audio_chunk_float32 = audio_chunk_float32.flatten()
        elif audio_chunk_float32.ndim > 1:
            print("VAD Warning: Received stereo audio, averaging to mono.")
            audio_chunk_float32 = np.mean(audio_chunk_float32, axis=1)

        if np.max(np.abs(audio_chunk_float32)) > 1.0:
            audio_chunk_float32 = np.clip(audio_chunk_float32, -1.0, 1.0)

        audio_chunk_int16 = (audio_chunk_float32 * 32767).astype(np.int16)

        num_frames = len(audio_chunk_int16) // self.frame_length
        if num_frames == 0 and len(audio_chunk_int16) > 0:
            padding = np.zeros(
                self.frame_length - len(audio_chunk_int16), dtype=np.int16
            )
            audio_chunk_int16 = np.concatenate((audio_chunk_int16, padding))
            num_frames = 1
        elif num_frames == 0 and len(audio_chunk_int16) == 0:
            return False

        try:
            for i in range(num_frames):
                start = i * self.frame_length
                end = start + self.frame_length
                frame = audio_chunk_int16[start:end]
                if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                    return True
            return False
        except Exception as e:
            print(
                f"VAD processing error: {e}. Chunk length: {len(audio_chunk_int16)}, Frame length: {self.frame_length}"
            )
            return False
