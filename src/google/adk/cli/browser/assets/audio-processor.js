/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.targetSampleRate = 16000;  // Live API expects 16 kHz PCM input
        this.originalSampleRate = sampleRate; // Browser's sample rate
        this.resampleRatio = this.originalSampleRate / this.targetSampleRate;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            let audioData = input[0]; // Get first channel's data
            
            if (this.resampleRatio !== 1) {
                audioData = this.resample(audioData);
            }

            this.port.postMessage(audioData);
        }
        return true; // Keep processor alive
    }

    resample(audioData) {
        const newLength = Math.round(audioData.length / this.resampleRatio);
        const resampled = new Float32Array(newLength);

        // Linear interpolation resampling (higher quality than nearest neighbor)
        const lastIndex = audioData.length - 1;
        for (let i = 0; i < newLength; i++) {
            const srcPos = i * this.resampleRatio;
            const srcIndex = Math.floor(srcPos);
            const nextIndex = Math.min(srcIndex + 1, lastIndex);
            const frac = srcPos - srcIndex;
            resampled[i] =
                audioData[srcIndex] * (1 - frac) + audioData[nextIndex] * frac;
        }
        return resampled;
    }
}

registerProcessor('audio-processor', AudioProcessor);
