#!/usr/bin/env python3
"""Standalone connectivity test for the AssemblyAI Voice Agent provider.

Requires ASSEMBLYAI_API_KEY and network access. Exits 0 on success, 1 on failure.

Usage:
    uv run src/tau2/voice/audio_native/assemblyai/test_provider_standalone.py
"""

import asyncio
import os
import signal
import sys

sys.path.insert(0, "src")

AUDIO_PATH = "tests/test_voice/test_audio_native/testdata/hello.ulaw"


async def main() -> int:
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        print("SKIP: ASSEMBLYAI_API_KEY not set")
        return 0

    from tau2.voice.audio_native.assemblyai.events import AAITimeoutEvent
    from tau2.voice.audio_native.assemblyai.provider import (
        AssemblyAIVADConfig,
        AssemblyAIVoiceAgentProvider,
    )

    provider = AssemblyAIVoiceAgentProvider()
    received = []
    try:
        print("1. Connecting...")
        await provider.connect()

        # session.ready is only emitted after we send session.update, so the
        # handshake (and session_id) completes inside configure_session.
        print("2. Configuring session (sends session.update, awaits ready)...")
        await provider.configure_session(
            system_prompt="You are a helpful assistant. Keep replies to one short sentence.",
            tools=[],
            vad_config=AssemblyAIVADConfig(),
        )
        assert provider.session_id, "no session_id after handshake"
        print(f"   session_id={provider.session_id}")

        print("3. Sending μ-law speech + 1s silence...")
        with open(AUDIO_PATH, "rb") as f:
            speech = f.read()
        await provider.send_audio(speech)
        await provider.send_audio(b"\xff" * 8000)  # 1s μ-law silence @ 8kHz

        print("4. Collecting events (up to 20s)...")
        for _ in range(200):
            events = await provider.receive_events_for_duration(0.1)
            for ev in events:
                if not isinstance(ev, AAITimeoutEvent):
                    received.append(ev)
            if len(received) >= 3:
                break

        types = sorted({type(e).__name__ for e in received})
        print(f"   received {len(received)} events: {types}")
        if received:
            print("✅ SUCCESS")
            return 0
        print("❌ FAILED — no events received")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        await provider.disconnect()


if __name__ == "__main__":

    def _timeout(signum, frame):
        print("\n❌ SCRIPT TIMEOUT (60s)")
        sys.exit(1)

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(60)
    sys.exit(asyncio.run(main()))
