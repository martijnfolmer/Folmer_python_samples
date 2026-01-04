from pydub import AudioSegment
import os
import warnings

"""
This script trims a WAV audio file to a specified duration and saves
the shortened result as a new WAV file.

Workflow:
- Load a WAV audio file using pydub
- Validate start position and duration
- Slice the audio from start_ms to start_ms + duration_ms
- Export the trimmed audio in WAV format
"""


def shorten_wav(input_file, output_file, start_ms, duration_ms):
    # Check if input file exists
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input audio file not found: {input_file}")

    # Load the audio file
    audio = AudioSegment.from_wav(input_file)
    audio_length_ms = len(audio)

    # Validate start_ms
    if start_ms < 0 or start_ms >= audio_length_ms:
        raise ValueError(
            f"start_ms ({start_ms}) is outside the audio length "
            f"(0–{audio_length_ms - 1} ms)"
        )

    end_ms = start_ms + duration_ms

    # Warn if requested duration exceeds audio length
    if end_ms > audio_length_ms:
        warnings.warn(
            f"Requested end_ms ({end_ms}) exceeds audio length "
            f"({audio_length_ms} ms). Trimming to end of file."
        )
        end_ms = audio_length_ms

    # Trim the audio
    shortened_audio = audio[start_ms:end_ms]

    # Export the shortened audio
    shortened_audio.export(output_file, format="wav")
    print(f"Shortened audio saved as: {output_file}")


# Example usage
input_wav = "AudioInputPath"    # Replace with your input file
output_wav = "AudioOutputPath"  # Replace with your desired output file
start_ms = 0                    # Where to start trimming
duration = 600                  # Duration in milliseconds

shorten_wav(input_wav, output_wav, start_ms, duration)
