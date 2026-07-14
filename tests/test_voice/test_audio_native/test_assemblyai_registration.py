from tau2.config import (
    AUDIO_NATIVE_PROVIDER_TYPES,
    DEFAULT_AUDIO_NATIVE_MODELS,
    DEFAULT_AUDIO_NATIVE_REASONING_EFFORT,
)
from tau2.data_model.simulation import AudioNativeConfig
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.assemblyai.discrete_time_adapter import (
    DiscreteTimeAssemblyAIAdapter,
)


def test_registry_entries_present():
    assert DEFAULT_AUDIO_NATIVE_MODELS["assemblyai"] == "managed"
    assert DEFAULT_AUDIO_NATIVE_REASONING_EFFORT["assemblyai"] is None
    assert AUDIO_NATIVE_PROVIDER_TYPES["assemblyai"] == "audio_native"


def test_config_accepts_provider():
    cfg = AudioNativeConfig(provider="assemblyai")
    assert cfg.provider == "assemblyai"


def test_factory_builds_adapter_without_connecting(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    adapter, model = create_adapter(
        provider="assemblyai", tick_duration_ms=200, model=None
    )
    assert isinstance(adapter, DiscreteTimeAssemblyAIAdapter)
    assert model == "managed"
    assert adapter.is_connected is False
