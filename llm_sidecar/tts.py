import torch
import torchaudio
from chatterbox import TTS
import io
import base64
import asyncio
from .event_bus import EventBus  # Assuming event_bus.py is in the same directory


class ChatterboxTTS:
    def __init__(self, model_dir, device, event_bus: EventBus):
        self.model = TTS.from_pretrained(model_dir).to(device)
        self.device = device
        self.event_bus = event_bus

    async def synth(self, text, ref_wav=None, exaggeration=0.5):
        if ref_wav:
            # This part still needs careful handling if ref_wav is bytes vs path
            # For now, assuming it's a path as per original Chatterbox usage
            ref_speech = self.model.get_ref_speech(ref_wav)
        else:
            ref_speech = None

        wav = self.model.synthesise(
            text,
            ref_speech=ref_speech,
            exaggeration_factor=exaggeration,
        )

        # Convert to raw WAV audio data as bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.cpu().unsqueeze(0), self.model.sr, format="wav")
        audio_bytes = buffer.getvalue()

        # Publish to Redis
        if self.event_bus:
            try:
                b64_audio_data = base64.b64encode(audio_bytes).decode("utf-8")
                await self.event_bus.publish("audio.bytes", b64_audio_data)
                print(
                    f"[TTS] Published {len(audio_bytes)} bytes of audio to 'audio.bytes' (b64 encoded)"
                )
            except Exception as e:
                print(f"[TTS] Error publishing to Redis: {e}")

        return audio_bytes
