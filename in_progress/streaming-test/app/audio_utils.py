import os
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as write_wav # To save audio if needed, or pass bytes directly
import io
import time

from google.cloud import speech
from google.cloud import texttospeech

from app.config import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS

# --- Audio Recording ---
def record_audio(duration_seconds: int, sample_rate: int = AUDIO_SAMPLE_RATE, channels: int = AUDIO_CHANNELS) -> bytes:
    """
    Records audio from the default microphone for a specified duration.

    Args:
        duration_seconds: The number of seconds to record.
        sample_rate: The recording sample rate.
        channels: The number of recording channels.

    Returns:
        A bytes object containing the recorded audio data in WAV format.
    """
    print(f"Recording audio for {duration_seconds} seconds at {sample_rate} Hz...")
    try:
        recording = sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Convert to WAV bytes
        byte_io = io.BytesIO()
        write_wav(byte_io, sample_rate, recording)
        return byte_io.getvalue()
    except Exception as e:
        print(f"Error during audio recording: {e}")
        print("Make sure you have a microphone connected and sounddevice is correctly installed.")
        print("You might need to install system dependencies for PortAudio (used by sounddevice).")
        print("On Debian/Ubuntu: sudo apt-get install libportaudio2")
        print("On Fedora: sudo dnf install portaudio-devel")
        print("On macOS (using Homebrew): brew install portaudio")
        return None

# --- Live Transcription (Speech-to-Text) ---
def transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int = AUDIO_SAMPLE_RATE) -> str:
    """
    Transcribes the given audio data using Google Cloud Speech-to-Text.

    Args:
        audio_bytes: Bytes object containing WAV audio data.
        sample_rate: The sample rate of the audio.

    Returns:
        The transcribed text, or None if transcription fails.
    """
    if not audio_bytes:
        print("No audio bytes provided for transcription.")
        return None

    print("Transcribing audio...")
    try:
        client = speech.SpeechClient()

        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # WAV is typically LINEAR16
            sample_rate_hertz=sample_rate,
            language_code="en-US",  # Adjust as needed
            # model="telephony" or "medical" or other specialized models can be used.
            # enable_automatic_punctuation=True, # Useful for more natural text
        )

        response = client.recognize(config=config, audio=audio)

        if response.results and response.results[0].alternatives:
            transcript = response.results[0].alternatives[0].transcript
            print(f"Transcription: {transcript}")
            return transcript
        else:
            print("No transcription results found.")
            return "" # Return empty string if no results
    except Exception as e:
        print(f"Error during audio transcription: {e}")
        return None

# --- Speech Synthesis (Text-to-Speech) ---
def synthesize_text_to_audio_bytes(text: str) -> bytes:
    """
    Synthesizes speech from the given text using Google Cloud Text-to-Speech.

    Args:
        text: The text to synthesize.

    Returns:
        A bytes object containing the synthesized audio data (e.g., MP3), or None if synthesis fails.
    """
    if not text:
        print("No text provided for synthesis.")
        return None

    print(f"Synthesizing audio for text: '{text[:50]}...'")
    try:
        client = texttospeech.TextToSpeechClient()

        input_text = texttospeech.SynthesisInput(text=text)

        # Configure the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",  # Adjust as needed
            # name="en-US-Wavenet-D", # Example Wavenet voice
            # For a list of voices: client.list_voices()
            # Standard voices are cheaper than Wavenet voices.
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL, # Or FEMALE, MALE
        )

        # Select the type of audio file you want
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3  # Or LINEAR16 for WAV
        )

        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        print("Audio synthesis finished.")
        return response.audio_content
    except Exception as e:
        print(f"Error during audio synthesis: {e}")
        return None

# --- Audio Playback ---
def play_audio_bytes(audio_bytes: bytes, sample_rate: int = AUDIO_SAMPLE_RATE, output_format="mp3"):
    """
    Plays back audio bytes.
    Note: sounddevice.play directly plays numpy arrays (PCM data).
          If we have MP3 or other formats, we'd need to decode them first.
          For simplicity, this example assumes we can get PCM if needed,
          or the system has a way to play MP3s (which `sd.play` does not do directly for MP3 bytes).

          This function is a placeholder and might need a more robust playback solution
          depending on the `audio_config.audio_encoding` used in synthesis.
          If LINEAR16 (WAV) is used in synthesis, it's more straightforward.
    Args:
        audio_bytes: Bytes object containing audio data.
        sample_rate: The sample rate for playback (relevant for PCM data).
        output_format: The format of audio_bytes (e.g., "mp3", "wav").
    """
    if not audio_bytes:
        print("No audio bytes provided for playback.")
        return

    print(f"Playing audio (format: {output_format})...")
    try:
        if output_format.lower() == "mp3":
            # sounddevice doesn't directly play MP3 bytes.
            # You'd typically need a library like 'pydub' or 'mpg123'/'ffplay' subprocess to play MP3s.
            # For this example, we'll print a message.
            # In a full app, you'd integrate a proper MP3 player.
            print("Playback of MP3 bytes requires an external player or library like pydub.")
            print("To actually play, you would save to a temp file and use a system command,")
            print("or use a library that can decode MP3 to PCM for sounddevice.")
            # Example with pydub (if installed and ffmpeg is available):
            # from pydub import AudioSegment
            # from pydub.playback import play
            # song = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            # play(song)
            # For now, just simulate playback time for non-PCM
            time.sleep(2) # Simulate playback

        elif output_format.lower() == "wav": # Assuming it's PCM data if WAV
            # If audio_bytes is raw PCM (e.g., from LINEAR16 synthesis decoded)
            # This needs to be actual PCM data, not WAV file bytes with headers.
            # For simplicity, assuming if it's 'wav', it's PCM data that sounddevice can handle.
            # This part needs careful handling of audio formats.
            # If it's a full WAV file in bytes, we need to parse it.
            from scipy.io.wavfile import read as read_wav
            byte_io = io.BytesIO(audio_bytes)
            rate, data = read_wav(byte_io) # This reads a WAV file, not raw PCM.
            sd.play(data, samplerate=rate)
            sd.wait()
        else:
            print(f"Unsupported audio format for direct playback: {output_format}")

        print("Playback finished.")
    except Exception as e:
        print(f"Error during audio playback: {e}")
        print("Ensure you have a speaker/audio output device.")
        print("For MP3 playback, consider installing 'pydub' and 'ffmpeg'.")
        print("For WAV playback with sounddevice, ensure the data is in a compatible PCM format.")


if __name__ == '__main__':
    # Basic test functions (will require manual interaction or pre-recorded files if run in non-interactive env)
    print("Testing audio_utils.py...")

    # 1. Record or Load Audio (Skipping live recording in automated subtask)
    # print("\n--- Testing Recording (manual check needed) ---")
    # recorded_audio_wav_bytes = record_audio(duration_seconds=3)
    # if recorded_audio_wav_bytes:
    #     with open("test_recording.wav", "wb") as f:
    #         f.write(recorded_audio_wav_bytes)
    #     print("Test recording saved to test_recording.wav (if microphone worked)")
    # else:
    #     print("Skipping recording test or it failed.")

    # For automated testing, let's create a dummy WAV file to use for transcription
    sample_rate_test = 16000
    duration_test = 1 # 1 second
    frequency_test = 440 # A4 note
    t_test = np.linspace(0, duration_test, int(sample_rate_test * duration_test), False)
    audio_data_test = (0.5 * np.sin(2 * np.pi * frequency_test * t_test) * 32767).astype(np.int16) # Sine wave
    dummy_wav_bytes_io = io.BytesIO()
    write_wav(dummy_wav_bytes_io, sample_rate_test, audio_data_test)
    dummy_wav_bytes = dummy_wav_bytes_io.getvalue()
    with open("dummy_audio_for_transcription.wav", "wb") as f:
        f.write(dummy_wav_bytes)
    print("Created dummy_audio_for_transcription.wav for testing.")


    # 2. Transcribe Audio
    print("\n--- Testing Transcription ---")
    # Prerequisite: GOOGLE_APPLICATION_CREDENTIALS must be set in the environment
    # and the associated account must have Speech-to-Text API enabled.
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and dummy_wav_bytes:
        transcribed_text = transcribe_audio_bytes(dummy_wav_bytes, sample_rate=sample_rate_test)
        if transcribed_text is not None:
            print(f"Test Transcription Result: '{transcribed_text}' (Note: dummy audio will likely result in empty or random transcription)")
        else:
            print("Transcription test failed or returned None.")
    else:
        print("Skipping transcription test: GOOGLE_APPLICATION_CREDENTIALS not set or no dummy audio.")

    # 3. Synthesize Text
    print("\n--- Testing Synthesis ---")
    # Prerequisite: GOOGLE_APPLICATION_CREDENTIALS must be set.
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        test_text_to_synthesize = "Hello, this is a test of the text to speech system."
        synthesized_audio_mp3_bytes = synthesize_text_to_audio_bytes(test_text_to_synthesize)
        if synthesized_audio_mp3_bytes:
            with open("test_synthesis.mp3", "wb") as f:
                f.write(synthesized_audio_mp3_bytes)
            print("Test synthesis saved to test_synthesis.mp3")

            # 4. Play Synthesized Audio (Placeholder in automated subtask)
            print("\n--- Testing Playback (manual check needed for actual sound) ---")
            play_audio_bytes(synthesized_audio_mp3_bytes, output_format="mp3")
            # To test WAV playback (requires synthesized_audio_bytes to be WAV)
            # synthesized_audio_wav_bytes = synthesize_text_to_audio_bytes(test_text_to_synthesize, audio_format=texttospeech.AudioEncoding.LINEAR16)
            # play_audio_bytes(synthesized_audio_wav_bytes, output_format="wav", sample_rate=...) # Need to know sample rate from TTS
        else:
            print("Synthesis test failed or returned None.")
    else:
        print("Skipping synthesis and playback test: GOOGLE_APPLICATION_CREDENTIALS not set.")

    print("\nAudio utils testing finished.")
