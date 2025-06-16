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

export class AudioStreamer {
    constructor(audioContext) {
      this.context = audioContext;
      this.sampleRate = 24000; // Output sample rate as per API spec
      this.audioQueue = [];
      this.isPlaying = false;
      this.currentSource = null;
      this.gainNode = this.context.createGain();
      this.gainNode.connect(this.context.destination);
      this.addPCM16 = this.addPCM16.bind(this);
      this.onComplete = () => {};
      this.playbackTimeout = null;
      this.lastPlaybackTime = 0;
    }

    addPCM16(chunk) {
      // Convert incoming PCM16 data to float32
      const float32Array = new Float32Array(chunk.length / 2);
      const dataView = new DataView(chunk.buffer);

      for (let i = 0; i < chunk.length / 2; i++) {
        try {
          const int16 = dataView.getInt16(i * 2, true);
          float32Array[i] = int16 / 32768;
        } catch (e) {
          console.error(e);
        }
      }

      // Create and fill audio buffer
      const audioBuffer = this.context.createBuffer(1, float32Array.length, this.sampleRate);
      audioBuffer.getChannelData(0).set(float32Array);

      // Add to queue and start playing if needed
      this.audioQueue.push(audioBuffer);
      
      if (!this.isPlaying) {
        this.isPlaying = true;
        this.lastPlaybackTime = this.context.currentTime;
        this.playNextBuffer();
      }

      // Ensure playback continues if it was interrupted
      this.checkPlaybackStatus();
    }

    checkPlaybackStatus() {
      // Clear any existing timeout
      if (this.playbackTimeout) {
        clearTimeout(this.playbackTimeout);
      }

      // Set a new timeout to check playback status
      this.playbackTimeout = setTimeout(() => {
        const now = this.context.currentTime;
        const timeSinceLastPlayback = now - this.lastPlaybackTime;

        // If more than 1 second has passed since last playback and we have buffers to play
        if (timeSinceLastPlayback > 1 && this.audioQueue.length > 0 && this.isPlaying) {
          console.log('Playback appears to have stalled, restarting...');
          this.playNextBuffer();
        }

        // Continue checking if we're still playing
        if (this.isPlaying) {
          this.checkPlaybackStatus();
        }
      }, 1000);
    }

    playNextBuffer() {
      if (this.audioQueue.length === 0) {
        this.isPlaying = false;
        return;
      }

      // Update last playback time
      this.lastPlaybackTime = this.context.currentTime;

      try {
        const audioBuffer = this.audioQueue.shift();
        const source = this.context.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.gainNode);

        // Store current source for potential stopping
        if (this.currentSource) {
          try {
            this.currentSource.disconnect();
          } catch (e) {
            // Ignore disconnection errors
          }
        }
        this.currentSource = source;

        // When this buffer ends, play the next one
        source.onended = () => {
          this.lastPlaybackTime = this.context.currentTime;
          if (this.audioQueue.length > 0) {
            // Small delay to ensure smooth transition
            setTimeout(() => this.playNextBuffer(), 0);
          } else {
            this.isPlaying = false;
            this.onComplete();
          }
        };

        // Start playing immediately
        source.start(0);
      } catch (error) {
        console.error('Error during playback:', error);
        // Try to recover by playing next buffer
        if (this.audioQueue.length > 0) {
          setTimeout(() => this.playNextBuffer(), 100);
        } else {
          this.isPlaying = false;
        }
      }
    }

    stop() {
      this.isPlaying = false;
      if (this.playbackTimeout) {
        clearTimeout(this.playbackTimeout);
        this.playbackTimeout = null;
      }
      if (this.currentSource) {
        try {
          this.currentSource.stop();
          this.currentSource.disconnect();
        } catch (e) {
          // Ignore if already stopped
        }
      }
      this.audioQueue = [];
      this.gainNode.gain.linearRampToValueAtTime(0, this.context.currentTime + 0.1);

      setTimeout(() => {
        this.gainNode.disconnect();
        this.gainNode = this.context.createGain();
        this.gainNode.connect(this.context.destination);
      }, 200);
    }

    async resume() {
      if (this.context.state === 'suspended') {
        await this.context.resume();
      }
      this.lastPlaybackTime = this.context.currentTime;
      this.gainNode.gain.setValueAtTime(1, this.context.currentTime);
      if (this.audioQueue.length > 0 && !this.isPlaying) {
        this.isPlaying = true;
        this.playNextBuffer();
      }
    }

    complete() {
      if (this.audioQueue.length > 0) {
        // Let the remaining buffers play out
        return;
      }
      if (this.playbackTimeout) {
        clearTimeout(this.playbackTimeout);
        this.playbackTimeout = null;
      }
      this.onComplete();
    }
  }