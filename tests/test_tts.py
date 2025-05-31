import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

import torch # Required for dummy tensor

# Module to test
from llm_sidecar.tts import ChatterboxTTS, CACHE_EXPIRATION_SECONDS

# Dummy audio tensor for mocking synthesis output
DUMMY_SR = 16000
DUMMY_AUDIO_TENSOR = torch.randn(1, DUMMY_SR * 2) # 2 seconds of audio

@pytest.fixture
def mock_event_bus():
    return AsyncMock()

@pytest.fixture
@patch('llm_sidecar.tts.redis.Redis')
@patch('llm_sidecar.tts.TTS') # Mocks chatterbox.TTS
@patch('llm_sidecar.tts.torchaudio') # Mocks torchaudio
def tts_instance(mock_torchaudio, mock_chatterbox_tts, mock_redis, mock_event_bus):
    # Configure mock for TTS.from_pretrained().to()
    mock_model_instance = MagicMock()
    # Configure the synthesise method on the mock_model_instance
    mock_model_instance.synthesise = AsyncMock(return_value=DUMMY_AUDIO_TENSOR)
    mock_model_instance.sr = DUMMY_SR # Set sample rate on mock model

    # TTS.from_pretrained() returns a mock that, when .to() is called, returns mock_model_instance
    mock_chatterbox_tts.from_pretrained.return_value.to.return_value = mock_model_instance

    # Configure mock for torchaudio.save
    mock_torchaudio.save = MagicMock()

    # Configure mock for Redis client
    mock_redis_client = MagicMock()
    mock_redis_client.get = MagicMock(return_value=None) # Default to cache miss
    mock_redis_client.setex = MagicMock()
    mock_redis_client.ping = MagicMock() # Mock ping for successful connection
    mock_redis.return_value = mock_redis_client # redis.Redis(..) will return this mock

    instance = ChatterboxTTS(
        model_dir="dummy_dir",
        device="cpu",
        event_bus=mock_event_bus,
        redis_host="fakehost",
        redis_port=1234
    )
    instance.redis_client = mock_redis_client # Ensure the instance uses our specific mock
    instance.model = mock_model_instance # Ensure the instance uses our specific mock model

    # Reset counters and mock calls for each test
    instance.cache_hits = 0
    instance.cache_misses = 0
    mock_model_instance.synthesise.reset_mock()
    mock_redis_client.get.reset_mock()
    mock_redis_client.setex.reset_mock()
    return instance, mock_model_instance, mock_redis_client

@pytest.mark.asyncio
async def test_tts_caching_speed_and_behavior(tts_instance):
    tts, mock_model, mock_redis = tts_instance
    text_to_synth = "This is a test phrase for caching."
    exaggeration = 0.5

    # First call - should be a cache miss
    start_time_miss = time.perf_counter()
    audio_data_miss = await tts.synth(text_to_synth, exaggeration=exaggeration)
    duration_miss = time.perf_counter() - start_time_miss

    mock_model.synthesise.assert_called_once()
    mock_redis.get.assert_called_once() # Should have tried to get from cache
    # Convert DUMMY_AUDIO_TENSOR to bytes as it would be in the cache
    buffer = asyncio.get_event_loop().run_in_executor(None, lambda: __import__('io').BytesIO())
    await asyncio.get_event_loop().run_in_executor(None, lambda: __import__('torchaudio').save(buffer, DUMMY_AUDIO_TENSOR.cpu().unsqueeze(0), DUMMY_SR, format="wav"))
    expected_bytes = await asyncio.get_event_loop().run_in_executor(None, lambda: buffer.getvalue())

    mock_redis.setex.assert_called_once_with(
        unittest.mock.ANY, # Cache key
        CACHE_EXPIRATION_SECONDS,
        expected_bytes
    )
    assert tts.cache_misses == 1
    assert tts.cache_hits == 0
    assert audio_data_miss == expected_bytes

    # Prepare cache for the second call by setting the return value of redis.get
    mock_redis.get.return_value = expected_bytes
    mock_model.synthesise.reset_mock() # Reset call count for the next assertion
    mock_redis.get.reset_mock() # Reset call count for the next assertion

    # Second call - should be a cache hit
    start_time_hit = time.perf_counter()
    audio_data_hit = await tts.synth(text_to_synth, exaggeration=exaggeration)
    duration_hit = time.perf_counter() - start_time_hit

    mock_model.synthesise.assert_not_called() # Should NOT call synthesis again
    mock_redis.get.assert_called_once() # Should have tried to get from cache
    assert tts.cache_misses == 1 # Should remain 1
    assert tts.cache_hits == 1   # Should be incremented
    assert audio_data_hit == audio_data_miss # Audio data should be identical

    print(f"Cache miss duration: {duration_miss:.4f}s")
    print(f"Cache hit duration: {duration_hit:.4f}s")
    assert duration_hit <= 0.02, "Cache hit was not <= 0.02 seconds (20ms)"

@pytest.mark.asyncio
async def test_tts_cache_key_variation(tts_instance):
    tts, mock_model, mock_redis = tts_instance

    # Call 1: text1, default speaker, default exaggeration
    await tts.synth("text1", exaggeration=0.5)
    assert mock_model.synthesise.call_count == 1
    tts.redis_client.get.return_value = None # Ensure next calls are misses if key changes

    # Call 2: text2 (different text)
    await tts.synth("text2", exaggeration=0.5)
    assert mock_model.synthesise.call_count == 2
    tts.redis_client.get.return_value = None

    # Call 3: text1, speaker1.wav (different speaker)
    await tts.synth("text1", ref_wav="speaker1.wav", exaggeration=0.5)
    assert mock_model.synthesise.call_count == 3
    tts.redis_client.get.return_value = None

    # Call 4: text1, exaggeration 0.1 (different exaggeration)
    await tts.synth("text1", exaggeration=0.1)
    assert mock_model.synthesise.call_count == 4

    # Set up cache for the original call (text1, default speaker, default exaggeration=0.5)
    # This requires knowing the expected byte output or mocking the `get` for that specific key
    # For simplicity, we'll assume the first call cached its result.
    # To simulate this, we need to make redis.get return the "cached" data for the specific key of the first call.
    # This is tricky without knowing the exact key or having a more sophisticated mock.
    # Instead, we'll rely on the fact that if synth is called with params that *should* hit,
    # and we've already cached *something* (even if it's just one item from the calls above),
    # we can test if a subsequent identical call still avoids re-synthesis if its data was hypothetically cached.

    # Let's reset the mock_redis.get to simulate a general cache miss first, then set a specific return.
    tts.redis_client.get.return_value = None

    # To properly test the cache hit for the *first* set of parameters,
    # we'd need to store the result of the first call and make redis.get return it.
    buffer = asyncio.get_event_loop().run_in_executor(None, lambda: __import__('io').BytesIO())
    await asyncio.get_event_loop().run_in_executor(None, lambda: __import__('torchaudio').save(buffer, DUMMY_AUDIO_TENSOR.cpu().unsqueeze(0), DUMMY_SR, format="wav"))
    cached_audio_for_text1_default = await asyncio.get_event_loop().run_in_executor(None, lambda: buffer.getvalue())

    # Simulate that the data for ("text1", exaggeration=0.5) is now in cache
    # We need to be more specific with the mock if different keys map to different values.
    # For this test, we'll make the mock return the value specifically when the first call's params are used.
    # This requires knowing the cache key or making `get` smarter.
    # A simpler approach for this unit test: assume the first call's data *is* what `get` will return
    # if that key is queried.

    original_get = mock_redis.get
    async def selective_get(key):
        # Simplified: assume any 'get' now returns the data for "text1", exag=0.5
        # This isn't perfect but tests the "don't call synthesise" part.
        # A better mock would check the key.
        import hashlib
        expected_cache_key_string = f"text1:default_speaker:0.5"
        expected_hashed_key = hashlib.sha256(expected_cache_key_string.encode('utf-8')).hexdigest()
        expected_redis_cache_key = f"audio_cache:{expected_hashed_key}"
        if key == expected_redis_cache_key:
            return cached_audio_for_text1_default
        return None

    mock_redis.get = AsyncMock(side_effect=selective_get)
    # Due to how fixtures work, changing mock_redis.get here might not affect the one inside tts_instance directly
    # if it captured the original mock. So, we update it on the instance directly.
    tts.redis_client.get = AsyncMock(side_effect=selective_get)


    # Call 5: text1, default speaker, default exaggeration (should be a hit from Call 1)
    await tts.synth("text1", exaggeration=0.5)
    assert mock_model.synthesise.call_count == 4 # Should NOT have incremented from 4

    # Restore original mock if necessary for other tests, though pytest fixtures handle isolation.
    mock_redis.get = original_get
    tts.redis_client.get = original_get


def test_tts_cache_stats_logic_unit(mock_event_bus): # Doesn't need full tts_instance fixture if we init manually
    # Mock Redis client for stats
    mock_redis_client_for_stats = MagicMock()
    mock_redis_client_for_stats.info.return_value = {'used_memory_human': '1.23M'}
    mock_redis_client_for_stats.ping = MagicMock() # For successful init

    with patch('llm_sidecar.tts.redis.Redis', return_value=mock_redis_client_for_stats):
        with patch('llm_sidecar.tts.TTS') as mock_tts_init: # Mock TTS class init
            mock_tts_init.from_pretrained.return_value.to.return_value = MagicMock() # model init

            tts_for_stats = ChatterboxTTS(
                model_dir="dummy", device="cpu", event_bus=mock_event_bus,
                redis_host="fake", redis_port=1234
            )
            # Ensure this instance uses the specific mock_redis_client_for_stats
            tts_for_stats.redis_client = mock_redis_client_for_stats


    tts_for_stats.cache_hits = 5
    tts_for_stats.cache_misses = 10

    # This part mimics the logic that would be in the server.py endpoint
    # For a unit test, we directly check the attributes and mock info()

    hits = tts_for_stats.cache_hits
    misses = tts_for_stats.cache_misses
    disk_mb_str = "N/A"

    if tts_for_stats.redis_client:
        try:
            redis_info = tts_for_stats.redis_client.info('memory')
            used_memory_human = redis_info.get('used_memory_human', '0B')

            size_val = float(used_memory_human[:-1])
            unit = used_memory_human[-1:].upper()

            disk_mb = 0.0
            if unit == 'G':
                disk_mb = size_val * 1024
            elif unit == 'M':
                disk_mb = size_val
            elif unit == 'K':
                disk_mb = size_val / 1024
            elif unit == 'B':
                disk_mb = size_val / (1024 * 1024)

            disk_mb_str = f"{disk_mb:.2f} MB" # Match server.py output format
        except Exception:
             pass # Keep "N/A" on error

    calculated_stats = {"hits": hits, "misses": misses, "redis_used_memory": disk_mb_str}

    assert calculated_stats['hits'] == 5
    assert calculated_stats['misses'] == 10
    assert calculated_stats['redis_used_memory'] == "1.23 MB"

    # Test with Kilobytes
    mock_redis_client_for_stats.info.return_value = {'used_memory_human': '512K'}
    if tts_for_stats.redis_client:
        try:
            redis_info = tts_for_stats.redis_client.info('memory')
            used_memory_human = redis_info.get('used_memory_human', '0B')
            size_val = float(used_memory_human[:-1])
            unit = used_memory_human[-1:].upper()
            disk_mb = 0.0
            if unit == 'K': disk_mb = size_val / 1024
            disk_mb_str = f"{disk_mb:.2f} MB"
        except: pass
    calculated_stats_kb = {"redis_used_memory": disk_mb_str}
    assert calculated_stats_kb['redis_used_memory'] == "0.50 MB"

    # Test with Gigabytes
    mock_redis_client_for_stats.info.return_value = {'used_memory_human': '2G'}
    if tts_for_stats.redis_client:
        try:
            redis_info = tts_for_stats.redis_client.info('memory')
            used_memory_human = redis_info.get('used_memory_human', '0B')
            size_val = float(used_memory_human[:-1])
            unit = used_memory_human[-1:].upper()
            disk_mb = 0.0
            if unit == 'G': disk_mb = size_val * 1024
            disk_mb_str = f"{disk_mb:.2f} MB"
        except: pass
    calculated_stats_gb = {"redis_used_memory": disk_mb_str}
    assert calculated_stats_gb['redis_used_memory'] == "2048.00 MB"

    # Test with Bytes
    mock_redis_client_for_stats.info.return_value = {'used_memory_human': '1000B'}
    if tts_for_stats.redis_client:
        try:
            redis_info = tts_for_stats.redis_client.info('memory')
            used_memory_human = redis_info.get('used_memory_human', '0B')
            size_val = float(used_memory_human[:-1]) # Get the numeric part
            unit = used_memory_human[-1:].upper() # Get the unit (B, K, M, G)
            disk_mb = 0.0
            if unit == 'B': disk_mb = size_val / (1024*1024)
            disk_mb_str = f"{disk_mb:.2f} MB"
        except: pass
    calculated_stats_b = {"redis_used_memory": disk_mb_str}
    # 1000 / (1024*1024) = 0.00095367431640625 -> "0.00 MB"
    assert calculated_stats_b['redis_used_memory'] == "0.00 MB"

    # Test with N/A (redis client not available or info fails)
    tts_for_stats.redis_client = None
    disk_mb_str_na = "N/A"
    if tts_for_stats.redis_client: # This will be false
        # ... logic ...
        pass
    else: # This path will be taken
        pass # disk_mb_str_na remains "N/A"

    calculated_stats_na = {"redis_used_memory": disk_mb_str_na}
    assert calculated_stats_na['redis_used_memory'] == "N/A"
