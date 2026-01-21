# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import sounddevice as sd

# input audio example. replace with the input audio you want to test
FILE_PATH = 'adk_live_audio_storage_input_audio_1762910896736.pcm'
# output audio example. replace with the input audio you want to test
FILE_PATH = 'adk_live_audio_storage_output_audio_1762910893258.pcm;rate=24000'
# PCM rate is always 24,000 for input and output
SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = np.int16  # Common types: int16, float32

# Read and play
with open(FILE_PATH, 'rb') as f:
  # Load raw data into numpy array
  raw_data = f.read()
  audio_array = np.frombuffer(raw_data, dtype=DTYPE)

  # Reshape if stereo (interleaved)
  if CHANNELS > 1:
    audio_array = audio_array.reshape((-1, CHANNELS))

  # Play
  print('Playing...')
  sd.play(audio_array, SAMPLE_RATE)
  sd.wait()
