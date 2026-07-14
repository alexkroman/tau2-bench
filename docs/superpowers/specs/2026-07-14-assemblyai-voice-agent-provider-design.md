# AssemblyAI Voice Agent audio-native provider for tau2-bench

**Date:** 2026-07-14
**Status:** Approved — ready for implementation planning

## Goal

Add an `assemblyai` provider to tau2-bench's voice full-duplex (audio-native)
evaluation mode so the **AssemblyAI Voice Agent API**
(`wss://agents.assemblyai.com/v1/ws`) can be benchmarked end-to-end on the
existing domains (airline, retail, telecom, banking_knowledge, mock).

### Scope decisions (locked)

- **What we evaluate (v1): AssemblyAI's managed model.** The Voice Agent API
  runs on AssemblyAI's own managed conversational LLM by default (STT + LLM +
  TTS + turn detection as one hosted stack). We do **not** wire up the optional
  "bring your own OpenAI-compatible LLM" (`llm`) field in v1. The model is
  therefore *endpoint-determined*, exactly like the existing `xai` provider.
- **Deliverable: working adapter + smoke test.** Full provider implementation,
  factory/config/CLI registration, and a standalone connectivity smoke test.
  Enough to run `tau2 run --audio-native --audio-native-provider assemblyai`.
- Out of scope for v1: BYO-LLM config, integration into the shared
  `test_provider_suite.py`, and EU endpoint selection.

## Why the xAI provider is the template

AssemblyAI's Voice Agent API is OpenAI-Realtime-shaped, like xAI's Grok Voice
Agent API, and **both natively speak G.711 μ-law at 8 kHz (`audio/pcmu`)**.
tau2's synthesis pipeline already produces telephony-format μ-law audio, so —
as with xAI — **no audio conversion is needed** in either direction. The
`src/tau2/voice/audio_native/xai/` provider is the closest existing analogue and
should be copied and adapted rather than written from scratch.

## Architecture

The new provider subclasses `DiscreteTimeAdapter`
(`src/tau2/voice/audio_native/adapter.py`). The base class continues to own:
TickResult creation, audio buffering/capping, proportional transcript
distribution, barge-in buffer clearing, tool-result queuing, tick timing, and
cumulative state. The provider implements only the provider-specific pieces:
`_execute_tick()`, `_flush_pending_tool_results()`, and the
`BackgroundAsyncLoop`-based lifecycle (`connect`/`disconnect`/`run_tick`/
`is_connected`), mirroring the xAI adapter.

## New files (`src/tau2/voice/audio_native/assemblyai/`)

- **`provider.py`** — `AssemblyAIVoiceAgentProvider`:
  - `connect()` — open WS with `Authorization: Bearer $ASSEMBLYAI_API_KEY`,
    wait for `session.ready`, store `session_id`.
  - `configure_session(system_prompt, tools, vad_config, voice)` — send one
    `session.update` (see "Session configuration").
  - `send_audio(bytes)` — `{"type": "input.audio", "audio": "<base64>"}`.
  - `send_tool_result(call_id, result_str)` — `{"type": "tool.result",
    "call_id": ..., "result": "<JSON string>"}`. **No** `reply.create`
    follow-up (tool.result auto-fires the next reply).
  - `receive_events()` / `receive_events_for_duration(seconds)`.
  - Supporting types: `AssemblyAIAudioFormat` (default `pcmu`),
    `AssemblyAIVADConfig` (`vad_threshold`, `min_silence`, `max_silence`,
    `interrupt_response`).
- **`events.py`** — Pydantic models for each server event plus
  `parse_assemblyai_event()`, `AssemblyAITimeoutEvent`, `AssemblyAIUnknownEvent`.
- **`discrete_time_adapter.py`** — `DiscreteTimeAssemblyAIAdapter`.
- **`__init__.py`** — exports.
- **`test_provider_standalone.py`** — smoke test (modeled on `nova/`).

## Auth, session configuration, audio

- **URL:** `wss://agents.assemblyai.com/v1/ws`
- **Header:** `Authorization: Bearer $ASSEMBLYAI_API_KEY` — note the `Bearer`
  prefix, which differs from AssemblyAI's REST API (raw key, no prefix).
- **Audio:** `audio/pcmu` (G.711 μ-law, 8 kHz) on both `input.format` and
  `output.format`. No conversion.
- **Session update:** a single `session.update` sent *after* `session.ready`.
  This is the first update, so the immutable-after-first-update fields
  (`output.voice`, `output.format`) are still settable here. Payload:
  ```json
  {"type": "session.update", "session": {
    "system_prompt": "<domain policy>",
    "input":  {"format": {"encoding": "audio/pcmu"}, "turn_detection": { ... }},
    "output": {"voice": "ivy", "format": {"encoding": "audio/pcmu"}},
    "tools":  [ /* flat function defs: type/name/description/parameters */ ]
  }}
  ```
- **Greeting: omitted by default.** The `greeting` string is piped straight to
  TTS (it bypasses the LLM), so a canned greeting would corrupt the benchmark.
  The managed LLM produces the first turn. If verification shows the agent will
  not speak first without a greeting, expose an optional greeting knob.

## Event mapping → `TickResult`

| AssemblyAI server event | Adapter handling |
|---|---|
| `session.ready` | handshake; store `session_id` |
| `session.updated` | config confirmed |
| `reply.started` | new `reply_id` → set `_current_item_id` |
| `reply.audio` | base64 μ-law chunk → `agent_audio_chunks` (item = `reply_id`); `UtteranceTranscript.add_audio` |
| `transcript.agent` | full agent text → `UtteranceTranscript.add_transcript` for that reply |
| `reply.done` | turn complete; `status: "interrupted"` → discard pending tool results |
| `input.speech.started` | barge-in → `vad_events`, clear buffered agent audio, mark truncation |
| `input.speech.stopped` | `vad_events` |
| `transcript.user` / `transcript.user.delta` | user input transcript (debug/logging) |
| `tool.call` | `ToolCall(id=call_id, name, arguments)` → `result.tool_calls` (dict field is `arguments`, **not** `args`) |
| `session.error` | map codes (`UNAUTHORIZED`, `invalid_format`, `invalid_audio`, …) → log / raise |

### Two behavioral differences from xAI (handled explicitly)

1. **Agent transcript is one full `transcript.agent`, not deltas.** Set the
   full text once via `UtteranceTranscript.add_transcript` for the reply, rather
   than accumulating deltas.
2. **`tool.result` auto-fires the next reply.** `_flush_pending_tool_results`
   sends the tool result and does **not** send any `response.create` /
   `reply.create` (that would produce a duplicate reply).

## Registration & config

- **`src/tau2/config.py`:**
  - `DEFAULT_ASSEMBLYAI_VOICE_AGENT_URL = "wss://agents.assemblyai.com/v1/ws"`
  - `DEFAULT_ASSEMBLYAI_VOICE = "ivy"`
  - `DEFAULT_ASSEMBLYAI_MODEL = "managed"` (endpoint-determined)
  - Add `"assemblyai"` to `DEFAULT_AUDIO_NATIVE_MODELS`,
    `DEFAULT_AUDIO_NATIVE_REASONING_EFFORT` (`None`), and
    `AUDIO_NATIVE_PROVIDER_TYPES` (`"audio_native"`).
- **`src/tau2/voice/audio_native/adapter.py`:** add an
  `elif provider == "assemblyai"` branch to `create_adapter()`, and add
  `"assemblyai"` to `_PROVIDERS_WITH_ENDPOINT_DETERMINED_MODEL`.
- **`src/tau2/data_model/simulation.py`:** add `"assemblyai"` to the
  `AudioNativeConfig.provider` `Literal` (and any CLI `--audio-native-provider`
  choice list that is not derived from the Literal).
- **`.env.example`** and the voice README env-var table: `ASSEMBLYAI_API_KEY`.

## Testing

- **Standalone smoke test** (`test_provider_standalone.py`): connect, configure
  the session, push a short audio buffer (or silence), assert `session.ready`
  and that events flow, then disconnect. Skips cleanly when `ASSEMBLYAI_API_KEY`
  is unset.
- **End-to-end run** (manual, needs the tester's own ElevenLabs voice personas
  per `docs/voice-personas.md`):
  ```bash
  tau2 run --domain airline --audio-native \
    --audio-native-provider assemblyai \
    --num-tasks 1 --speech-complexity control --verbose-logs
  ```

## Open items to verify during implementation

1. **First turn with greeting omitted** — confirm the agent replies on the
   first user audio; if not, expose an optional greeting.
2. **`reply.audio` item/reply-id field name** — associate audio with the reply;
   fall back to `_current_item_id` set at `reply.started`.
3. **`--audio-native-provider` choice source** — whether CLI choices derive from
   the `Literal` or are a separate hardcoded list that must also be updated.

## Non-goals

- BYO OpenAI-compatible LLM (`llm` field).
- EU endpoint (`wss://agents.eu.assemblyai.com/v1/ws`).
- `session.resume` reconnection handling.
- Shared `test_provider_suite.py` integration and leaderboard submission.
