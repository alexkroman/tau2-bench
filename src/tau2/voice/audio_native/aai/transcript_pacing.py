"""Reveal aai agent text to the simulated caller at the pace it is SPOKEN.

The simulated caller does not listen to agent audio; it reads the text tau2
slices out of each reply in proportion to playback, and its interruption and
backchannel decisions are made on that text. The shared slicing
(`UtteranceTranscript.get_transcript_for_audio`) has two biases that matter for
this agent, which reads order ids and ZIPs digit by digit:

- The denominator is the audio received SO FAR, which grows while TTS streams,
  so text known ahead of its audio is revealed against too small a total: it
  runs ahead, then freezes mid-word for seconds while the ratio recovers
  ("…Suite 564, Ho" [2.8s] "uston, Texas"). Measured on a 114-task retail
  run: when the caller cut in, it had read a median 77% of the reply's text
  but heard only 57% of its audio, and 63 interruptions came after the whole
  sentence was on screen with a quarter or more of it still unspoken.
- Every character is weighted equally, but "W-5-0-5-6-5-1-9" is 15 characters
  and ~4s of spoken digits. In a reply that ends in an id or a ZIP, character
  slicing lags through the words and then races through the digits.

This lives in the aai package rather than in the shared slicing so that other
providers' runs are untouched; the subclass is what the aai adapter stores, and
the shared `get_proportional_transcript` reaches it through the overridden
method.
"""

from bisect import bisect_left
from itertools import accumulate

from tau2.voice.audio_native.tick_result import UtteranceTranscript

# Spoken length of one character, in letter-units. A digit is said as a word
# ("seven", "zero"). 5 is calibrated on the run above: it makes digit-heavy and
# digit-free replies pace at the same rate (14.3 vs 14.5 units/s), where equal
# weighting made digit-heavy text read 25% slow.
DIGIT_WEIGHT = 5
SYMBOL_WEIGHTS = {"$": 7, "%": 7}  # "dollars", "percent"

# Speaking rate in those units, measured on the same replies.
SPEECH_UNITS_PER_SECOND = 14.5


def char_weight(char: str) -> int:
    """Spoken length of one character in letter-units."""
    if char.isdigit():
        return DIGIT_WEIGHT
    return SYMBOL_WEIGHTS.get(char, 1)


def spoken_weight(text: str) -> int:
    """Estimated spoken length of ``text`` in letter-units."""
    return sum(char_weight(c) for c in text)


def paced_text(
    transcript: str, total_bytes: float, played_bytes: float, start_char: int
) -> tuple[str, int]:
    """The slice of ``transcript`` covered by ``played_bytes`` of audio.

    Same contract as the shared ``get_proportional_text`` — (text, new
    position), never moving backwards — with the played fraction applied to
    spoken length instead of character count. For plain letters the two agree.
    """
    if total_bytes <= 0 or not transcript:
        return "", start_char
    ratio = min(1.0, played_bytes / total_bytes)
    # cumulative[i] is the spoken weight of transcript[:i].
    cumulative = [0, *accumulate(char_weight(c) for c in transcript)]
    end_char = max(start_char, bisect_left(cumulative, ratio * cumulative[-1]))
    return transcript[start_char:end_char], end_char


class PacedUtteranceTranscript(UtteranceTranscript):
    """An `UtteranceTranscript` that reveals text at its spoken pace.

    `bytes_per_second` is the rate of the audio counted by `add_audio` (the
    adapter's mu-law 8k). Until `audio_complete`, the total is at least the
    text's estimated spoken duration, so text that arrives ahead of its audio
    is not revealed against a partial total.
    """

    bytes_per_second: float
    audio_complete: bool = False

    def expected_audio_bytes(self) -> float:
        """Best estimate of this utterance's total audio, in bytes."""
        if self.audio_complete:
            return self.audio_bytes_received
        estimated = (
            spoken_weight(self.transcript_received)
            / SPEECH_UNITS_PER_SECOND
            * self.bytes_per_second
        )
        return max(self.audio_bytes_received, estimated)

    def get_transcript_for_audio(self, audio_bytes_to_play: int) -> str:
        self.audio_bytes_played += audio_bytes_to_play
        if self.audio_bytes_received == 0 or not self.transcript_received:
            return ""
        text, self.transcript_chars_shown = paced_text(
            self.transcript_received,
            self.expected_audio_bytes(),
            self.audio_bytes_played,
            self.transcript_chars_shown,
        )
        return text
