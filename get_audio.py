import re
import sys
import queue
import subprocess as sp
from google.cloud import speech

# Audio parameters (must match FFmpeg output)
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class AudioStream:
    """Opens an audio stream from FFmpeg as a generator yielding audio chunks."""

    def __init__(self, video_url: str, rate: int = RATE, chunk: int = CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = False

        # Start FFmpeg process
        self._ffmpeg_process = (
            sp.Popen([
                'ffmpeg',
                '-i', video_url,
                '-f', 'wav',
                '-ac', '1',
                '-ar', str(rate),
                '-'
            ], stdout=sp.PIPE, stderr=sp.DEVNULL)
        )

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.closed = True
        if self._ffmpeg_process:
            self._ffmpeg_process.terminate()
            try:
                self._ffmpeg_process.wait(timeout=5)
            except sp.TimeoutExpired:
                self._ffmpeg_process.kill()

    def generator(self):
        """Generates audio chunks from FFmpeg's stdout"""
        while not self.closed:
            chunk = self._ffmpeg_process.stdout.read(self._chunk)
            if not chunk:
                break
            yield chunk


def transcribe_video_stream(video_url: str) -> str:
    """Transcribes audio from a video URL stream"""
    language_code = "en-US"

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    with AudioStream(video_url) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)
        return listen_print_loop(responses)  # Reuse your existing function


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <video_url>")
        sys.exit(1)

    video_url = sys.argv[1]
    transcribe_video_stream(video_url)