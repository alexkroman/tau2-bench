from tau2.voice.audio_native.assemblyai.events import (
    AAIAgentTranscriptEvent,
    AAIReplyAudioEvent,
    AAISessionReadyEvent,
    AAIToolCallEvent,
    AAIUnknownEvent,
    parse_assemblyai_event,
)


def test_parse_session_ready():
    ev = parse_assemblyai_event({"type": "session.ready", "session_id": "s-123"})
    assert isinstance(ev, AAISessionReadyEvent)
    assert ev.session_id == "s-123"


def test_parse_reply_audio_wire_format_uses_data():
    # Live API shape: base64 payload arrives under `data`.
    ev = parse_assemblyai_event(
        {"type": "reply.audio", "data": "QUJD", "reply_id": "r-1"}
    )
    assert isinstance(ev, AAIReplyAudioEvent)
    assert ev.audio == "QUJD"
    assert ev.reply_id == "r-1"


def test_parse_reply_audio_accepts_legacy_audio_field():
    ev = parse_assemblyai_event(
        {"type": "reply.audio", "audio": "QUJD", "reply_id": "r-1"}
    )
    assert isinstance(ev, AAIReplyAudioEvent)
    assert ev.audio == "QUJD"
    assert ev.reply_id == "r-1"


def test_parse_agent_transcript_interrupted():
    ev = parse_assemblyai_event(
        {
            "type": "transcript.agent",
            "text": "hello",
            "interrupted": True,
            "reply_id": "r-1",
        }
    )
    assert isinstance(ev, AAIAgentTranscriptEvent)
    assert ev.text == "hello"
    assert ev.interrupted is True


def test_parse_tool_call_wire_format_uses_args():
    # Live API shape: tool arguments arrive under `args`.
    ev = parse_assemblyai_event(
        {
            "type": "tool.call",
            "call_id": "c-1",
            "name": "lookup_order",
            "args": {"order_id": "ORD-1"},
        }
    )
    assert isinstance(ev, AAIToolCallEvent)
    assert ev.call_id == "c-1"
    assert ev.name == "lookup_order"
    assert ev.arguments == {"order_id": "ORD-1"}


def test_parse_tool_call_accepts_arguments_field():
    ev = parse_assemblyai_event(
        {
            "type": "tool.call",
            "call_id": "c-1",
            "name": "lookup_order",
            "arguments": {"order_id": "ORD-1"},
        }
    )
    assert isinstance(ev, AAIToolCallEvent)
    assert ev.arguments == {"order_id": "ORD-1"}


def test_parse_session_updated_wire_format_uses_config():
    ev = parse_assemblyai_event(
        {"type": "session.updated", "config": {"system_prompt": "hi", "tools": []}}
    )
    assert ev.type == "session.updated"
    assert ev.config == {"system_prompt": "hi", "tools": []}


def test_parse_unknown_event():
    ev = parse_assemblyai_event({"type": "does.not.exist", "foo": 1})
    assert isinstance(ev, AAIUnknownEvent)
    assert ev.raw == {"type": "does.not.exist", "foo": 1}
