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

import { audioContext } from "../utils/utils.js";
import { createWorkletFromSrc, registeredWorklets } from "./audioworklet-registry.js";
import AudioRecordingWorklet from "./audio-recording-worklet.js";

function arrayBufferToBase64(buffer) {
  var binary = "";
  var bytes = new Uint8Array(buffer);
  var len = bytes.byteLength;
  for (var i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

export class AudioRecorder extends EventEmitter3 {
  constructor() {
    super();
    this.sampleRate = 16000;
    this.stream = undefined;
    this.audioContext = undefined;
    this.source = undefined;
    this.recording = false;
    this.recordingWorklet = undefined;
    this.starting = null;
    this.isMuted = false;
  }

  async start() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Could not request user media");
    }

    this.starting = new Promise(async (resolve, reject) => {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.audioContext = await audioContext({ sampleRate: this.sampleRate });
      this.source = this.audioContext.createMediaStreamSource(this.stream);

      const workletName = "audio-recorder-worklet";
      const workletSrc = AudioRecordingWorklet;

      if (!registeredWorklets.has(this.audioContext)) {
        registeredWorklets.set(this.audioContext, {});
      }

      const registry = registeredWorklets.get(this.audioContext);
      if (!registry[workletName]) {
        const src = createWorkletFromSrc(workletName, workletSrc);
        await this.audioContext.audioWorklet.addModule(src);
        registry[workletName] = {
          node: new AudioWorkletNode(this.audioContext, workletName),
          handlers: []
        };
      }

      this.recordingWorklet = registry[workletName].node;

      this.recordingWorklet.port.onmessage = async (ev) => {
        const arrayBuffer = ev.data.data.int16arrayBuffer;

        if (arrayBuffer) {
          const arrayBufferString = arrayBufferToBase64(arrayBuffer);
          this.emit("data", arrayBufferString);
        }
      };
      this.source.connect(this.recordingWorklet);

      this.recording = true;
      resolve();
      this.starting = null;
    });
  }

  stop() {
    const handleStop = () => {
      this.source?.disconnect();
      this.stream?.getTracks().forEach((track) => track.stop());
      this.stream = undefined;
      this.recordingWorklet = undefined;
    };
    if (this.starting) {
      this.starting.then(handleStop);
      return;
    }
    handleStop();
  }

  mute() {
    if (this.source && this.recordingWorklet && !this.isMuted) {
      this.source.disconnect(this.recordingWorklet);
      this.isMuted = true;
    }
  }

  unmute() {
    if (this.source && this.recordingWorklet && this.isMuted) {
      this.source.connect(this.recordingWorklet);
      this.isMuted = false;
    }
  }
}