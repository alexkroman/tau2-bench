from tau2.environment.tool import Tool
from tau2.voice.audio_native.assemblyai.provider import (
    AssemblyAIAudioFormat,
    AssemblyAIVADConfig,
    AssemblyAIVoiceAgentProvider,
)


def _provider():
    return AssemblyAIVoiceAgentProvider(api_key="test-key")


def test_session_payload_sets_pcmu_and_voice_and_prompt():
    p = _provider()
    payload = p._build_session_payload(
        system_prompt="You are helpful.",
        tools=[],
        vad_config=AssemblyAIVADConfig(),
        voice="ivy",
    )
    assert payload["type"] == "session.update"
    s = payload["session"]
    assert s["system_prompt"] == "You are helpful."
    assert s["input"]["format"]["encoding"] == "audio/pcmu"
    assert s["output"]["format"]["encoding"] == "audio/pcmu"
    assert s["output"]["voice"] == "ivy"
    assert "greeting" not in s  # greeting deliberately omitted
    assert s["input"]["turn_detection"]["min_silence"] == 1000
    assert s["input"]["turn_detection"]["max_silence"] == 3000


def test_tools_use_flat_format():
    def lookup(order_id: str) -> str:
        """Look up an order.

        Args:
            order_id: The order id.
        """
        return order_id

    tools = [Tool(func=lookup)]
    p = _provider()
    formatted = p._format_tools_for_api(tools)
    assert formatted[0]["type"] == "function"
    assert formatted[0]["name"] == "lookup"
    assert "parameters" in formatted[0]
    assert "function" not in formatted[0]  # flat, not nested


def test_audio_format_default_is_pcmu():
    assert (
        AssemblyAIVoiceAgentProvider(api_key="k").audio_format
        == AssemblyAIAudioFormat.PCMU
    )


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    import pytest

    with pytest.raises(ValueError):
        AssemblyAIVoiceAgentProvider()
