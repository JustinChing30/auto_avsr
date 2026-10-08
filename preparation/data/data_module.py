#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import av
import numpy as np
import torch
import torchaudio
import torchvision


class AVSRDataLoader:
    def __init__(self, modality, detector="retinaface", convert_gray=True, gpu_type="cuda"):
        self.modality = modality
        if modality == "video":
            if detector == "retinaface":
                from detectors.retinaface.detector import LandmarksDetector
                from detectors.retinaface.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector(device=gpu_type+":0")
                self.video_process = VideoProcess(convert_gray=convert_gray)

            if detector == "mediapipe":
                from detectors.mediapipe.detector import LandmarksDetector
                from detectors.mediapipe.video_process import VideoProcess

                self.landmarks_detector = LandmarksDetector()
                self.video_process = VideoProcess(convert_gray=convert_gray)

    def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return audio
        if self.modality == "video":
            video = self.load_video(data_filename)
            if not landmarks:
                landmarks = self.landmarks_detector(video)
            video = self.video_process(video, landmarks)
            if video is None:
                raise TypeError("video cannot be None")
            video = torch.tensor(video)
            return video

    def load_audio(self, data_filename):
        container = av.open(data_filename)
        if len(container.streams.audio) == 0:
            container.close()
            return torch.zeros((1, 16000), dtype=torch.float32), 16000

        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.codec_context.sample_rate

        resampler = av.AudioResampling(
            format="fltp",
            layout="mono",
            rate=sample_rate,
        )

        frames = []
        for frame in container.decode(audio_stream):
            resampled = resampler.resample(frame)
            if resampled:
                frames.append(resampled[0].to_ndarray())

        container.close()

        if len(frames) == 0:
            return torch.zeros((1, 16000), dtype=torch.float32), 16000

        audio_np = np.concatenate(frames, axis=-1)
        if audio_np.ndim == 1:
            audio_np = np.expand_dims(audio_np, axis=0)

        waveform = torch.from_numpy(audio_np).to(torch.float32)
        return waveform, sample_rate

    def load_video(self, data_filename):
        container = av.open(data_filename)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        return np.stack(frames)

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, target_sample_rate
            )
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
