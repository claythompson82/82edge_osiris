import torch
import torchaudio
from chatterbox import TTS
import io
import base64
import asyncio
import redis # Added redis import
import hashlib # Added for caching
import os # Added for path manipulation for speaker name

from .event_bus import EventBus # Assuming event_bus.py is in the same directory

CACHE_EXPIRATION_SECONDS = 24 * 60 * 60 # 24 hours

class ChatterboxTTS:
    def __init__(self, model_dir, device, event_bus: EventBus, redis_host: str = 'localhost', redis_port: int = 6379):
        self.model = TTS.from_pretrained(model_dir).to(device)
        self.device = device
        self.event_bus = event_bus
        self.cache_hits = 0
        self.cache_misses = 0
        try:
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=False) # Important: decode_responses=False for bytes
            self.redis_client.ping() # Verify connection
            print(f"[TTS] Successfully connected to Redis at {redis_host}:{redis_port}")
        except redis.exceptions.ConnectionError as e:
            print(f"[TTS] Error connecting to Redis at {redis_host}:{redis_port}: {e}")
            self.redis_client = None # Set to None if connection fails
        except Exception as e: # Catch other potential errors like redis library not being installed
            print(f"[TTS] An unexpected error occurred during Redis initialization: {e}")
            self.redis_client = None


    async def synth(self, text, ref_wav=None, exaggeration=0.5):
        speaker_identifier = "default_speaker"
        if ref_wav and isinstance(ref_wav, str):
            try:
                speaker_identifier = os.path.basename(ref_wav)
            except Exception: # Handle potential errors with os.path.basename if ref_wav is not a valid path
                pass # Keep default_speaker

        cache_key_string = f"{text}:{speaker_identifier}:{str(exaggeration)}"
        hashed_key = hashlib.sha256(cache_key_string.encode('utf-8')).hexdigest()
        redis_cache_key = f"audio_cache:{hashed_key}"

        if self.redis_client:
            try:
                cached_audio = self.redis_client.get(redis_cache_key)
                if cached_audio:
                    self.cache_hits += 1
                    print(f"[TTS] Cache hit for key: {redis_cache_key}. Total hits: {self.cache_hits}, misses: {self.cache_misses}")
                    return cached_audio # Return bytes directly
            except redis.exceptions.RedisError as e:
                print(f"[TTS] Redis error during cache lookup: {e}")
                # Proceed to synthesis as if cache miss
            except Exception as e:
                print(f"[TTS] Unexpected error during cache lookup: {e}")


        self.cache_misses += 1
        print(f"[TTS] Cache miss for key: {redis_cache_key}. Total hits: {self.cache_hits}, misses: {self.cache_misses}")

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

        if self.redis_client:
            try:
                self.redis_client.setex(redis_cache_key, CACHE_EXPIRATION_SECONDS, audio_bytes)
                print(f"[TTS] Stored in cache with key: {redis_cache_key}")
            except redis.exceptions.RedisError as e:
                print(f"[TTS] Redis error during cache set: {e}")
            except Exception as e:
                print(f"[TTS] Unexpected error during cache set: {e}")

        # Publish to Redis via EventBus (if EventBus uses Redis)
        if self.event_bus:
            try:
                b64_audio_data = base64.b64encode(audio_bytes).decode('utf-8')
                # Assuming EventBus might handle the direct Redis publishing if configured
                await self.event_bus.publish("audio.bytes", b64_audio_data)
                print(f"[TTS] Published {len(audio_bytes)} bytes of audio via EventBus to 'audio.bytes' (b64 encoded)")
            except Exception as e:
                print(f"[TTS] Error publishing via EventBus: {e}")
        
        # Example: Direct publish to Redis (if needed, though EventBus is preferred)
        # if self.redis_client:
        #     try:
        #         b64_audio_data = base64.b64encode(audio_bytes).decode('utf-8')
        #         self.redis_client.publish("direct.audio.bytes", b64_audio_data)
        #         print(f"[TTS] Directly published {len(audio_bytes)} bytes of audio to 'direct.audio.bytes' (b64 encoded)")
        #     except Exception as e:
        #         print(f"[TTS] Error directly publishing to Redis: {e}")

        return audio_bytes
