import os

import librosa
import numpy as np
from openvino import Core
from sklearn.cluster import AgglomerativeClustering


class OpenVINODiarizer:
    """Speaker diarization with WeSpeaker inference on OpenVINO GPU."""

    def __init__(self, embed_model_path, device="GPU", sample_rate=16000,
                 distance_threshold=0.5, num_speakers=0,
                 max_auto_speakers=8,
                 window_seconds=2.0, hop_seconds=1.0):
        self.sample_rate = sample_rate
        self.device = device.upper()
        self.distance_threshold = float(distance_threshold)
        self.num_speakers = int(num_speakers)
        self.max_auto_speakers = max(2, int(max_auto_speakers))
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.target_frames = 200
        self.last_mode = "wespeaker_gpu"
        self.last_raw_cluster_count = 0
        self.last_cluster_count = 0
        self.core = Core()

        if self.device != "GPU":
            raise ValueError("Диаризация настроена только для GPU. Выберите устройство GPU.")
        if not os.path.isfile(embed_model_path):
            raise FileNotFoundError(f"Embedding-модель не найдена: {embed_model_path}")

        model = self.core.read_model(embed_model_path)
        if len(model.inputs) != 1:
            raise ValueError(f"Ожидался один вход embedding-модели, получено: {len(model.inputs)}")
        self.embed_compiled = self.core.compile_model(model, "GPU")
        self.embed_input = self.embed_compiled.input(0)
        self.embed_output = self.embed_compiled.output(0)

    def _log_mel_features(self, audio):
        """Build a fixed [1, 200, 80] feature tensor for WeSpeaker."""
        target_samples = int(self.window_seconds * self.sample_rate)
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        elif len(audio) > target_samples:
            audio = audio[:target_samples]

        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - 0.97 * audio[:-1]
        mel = librosa.feature.melspectrogram(
            y=emphasized,
            sr=self.sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            window="hamming",
            center=False,
            power=2.0,
            n_mels=80,
            fmin=20,
            fmax=7600,
            htk=True,
            norm=None,
        )
        features = np.log(np.maximum(mel, 1e-10)).T
        features -= features.mean(axis=0, keepdims=True)

        if len(features) < self.target_frames:
            features = np.pad(features, ((0, self.target_frames - len(features)), (0, 0)))
        else:
            features = features[:self.target_frames]
        return np.ascontiguousarray(features[np.newaxis], dtype=np.float32)

    def _embedding(self, audio):
        features = self._log_mel_features(audio)
        result = self.embed_compiled({self.embed_input: features})[self.embed_output]
        embedding = np.asarray(result, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        return embedding / norm

    def _make_windows(self, audio, speech_regions):
        window_samples = int(self.window_seconds * self.sample_rate)
        hop_samples = int(self.hop_seconds * self.sample_rate)
        duration = len(audio) / self.sample_rate
        windows = []

        for region_id, (start_sec, end_sec) in enumerate(speech_regions):
            start_sec = max(0.0, float(start_sec))
            end_sec = min(duration, float(end_sec))
            if end_sec - start_sec < 0.35:
                continue
            first = int(start_sec * self.sample_rate)
            last = int(end_sec * self.sample_rate)
            starts = list(range(first, max(first + 1, last - window_samples + 1), hop_samples))
            tail_start = max(first, last - window_samples)
            if not starts or tail_start - starts[-1] > hop_samples // 3:
                starts.append(tail_start)

            for sample_start in starts:
                sample_end = min(sample_start + window_samples, len(audio))
                chunk = audio[sample_start:sample_end]
                if len(chunk) < int(0.35 * self.sample_rate):
                    continue
                rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
                if rms < 1e-4:
                    continue
                windows.append({
                    "start": sample_start / self.sample_rate,
                    "end": sample_end / self.sample_rate,
                    "region": region_id,
                    "audio": chunk,
                })
        return windows

    def _cluster(self, embeddings):
        if len(embeddings) == 1:
            self.last_raw_cluster_count = 1
            self.last_cluster_count = 1
            return np.zeros(1, dtype=int)
        if self.num_speakers > 0:
            model = AgglomerativeClustering(
                n_clusters=min(self.num_speakers, len(embeddings)),
                metric="cosine",
                linkage="average",
            )
            labels = model.fit_predict(embeddings)
            self.last_raw_cluster_count = len(set(labels))
            self.last_cluster_count = self.last_raw_cluster_count
            return labels
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage="average",
            )
            labels = model.fit_predict(embeddings)
            self.last_raw_cluster_count = len(set(labels))
            if self.last_raw_cluster_count > self.max_auto_speakers:
                labels = AgglomerativeClustering(
                    n_clusters=min(self.max_auto_speakers, len(embeddings)),
                    metric="cosine",
                    linkage="average",
                ).fit_predict(embeddings)
            self.last_cluster_count = len(set(labels))
            return labels

    @staticmethod
    def _smooth_labels(windows, labels):
        labels = np.asarray(labels).copy()
        for i in range(1, len(labels) - 1):
            same_region = windows[i - 1]["region"] == windows[i]["region"] == windows[i + 1]["region"]
            if same_region and labels[i - 1] == labels[i + 1] != labels[i]:
                labels[i] = labels[i - 1]
        return labels

    @staticmethod
    def _windows_to_segments(windows, labels, speech_regions):
        segments = []
        for region_id, (region_start, region_end) in enumerate(speech_regions):
            indices = [i for i, window in enumerate(windows) if window["region"] == region_id]
            if not indices:
                continue
            region_windows = [windows[i] for i in indices]
            region_labels = [int(labels[i]) for i in indices]
            centers = [(w["start"] + w["end"]) / 2 for w in region_windows]
            boundaries = [float(region_start)]
            boundaries.extend((centers[i - 1] + centers[i]) / 2 for i in range(1, len(centers)))
            boundaries.append(float(region_end))

            for i, label in enumerate(region_labels):
                start, end = boundaries[i], boundaries[i + 1]
                speaker = f"SPEAKER_{label + 1:02d}"
                if segments and segments[-1][2] == speaker and start - segments[-1][1] <= 0.15:
                    segments[-1] = (segments[-1][0], end, speaker)
                else:
                    segments.append((start, end, speaker))
        return segments

    def diarize(self, audio, sample_rate=16000, speech_regions=None):
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=self.sample_rate, axis=-1
            )

        mono_audio = audio.mean(axis=0) if audio.ndim == 2 else audio
        duration = mono_audio.shape[-1] / self.sample_rate
        if not speech_regions:
            speech_regions = [(0.0, duration)]
        speech_regions = sorted(speech_regions)

        if audio.ndim == 2 and audio.shape[0] >= 2 and self._channels_are_isolated(audio, speech_regions):
            self.last_mode = "separate_channels"
            return self._diarize_separate_channels(audio, speech_regions)

        self.last_mode = "wespeaker_gpu"
        windows = self._make_windows(mono_audio, speech_regions)
        valid_windows = []
        embeddings = []
        for window in windows:
            embedding = self._embedding(window["audio"])
            if embedding is not None:
                valid_windows.append(window)
                embeddings.append(embedding)
        if not embeddings:
            return []

        labels = self._cluster(np.asarray(embeddings, dtype=np.float32))
        if self.num_speakers == 0:
            labels = self._smooth_labels(valid_windows, labels)
        return self._windows_to_segments(valid_windows, labels, speech_regions)

    def _channels_are_isolated(self, audio, speech_regions):
        """Detect telephone/meeting recordings with one speaker per channel."""
        first, second = audio[0], audio[1]
        correlation = float(np.corrcoef(first, second)[0, 1])
        dominance = []
        for start_sec, end_sec in speech_regions:
            start = int(start_sec * self.sample_rate)
            end = int(end_sec * self.sample_rate)
            if end - start < int(0.15 * self.sample_rate):
                continue
            power = np.mean(np.square(audio[:2, start:end]), axis=1) + 1e-12
            dominance.append(abs(10 * np.log10(power[0] / power[1])))
        return bool(dominance) and abs(correlation) < 0.25 and np.median(dominance) >= 6.0

    def _diarize_separate_channels(self, audio, speech_regions):
        segments = []
        channel_count = min(audio.shape[0], self.num_speakers or audio.shape[0])
        for start_sec, end_sec in speech_regions:
            start = max(0, int(start_sec * self.sample_rate))
            end = min(audio.shape[1], int(end_sec * self.sample_rate))
            if end <= start:
                continue
            power = np.mean(np.square(audio[:channel_count, start:end]), axis=1)
            channel = int(np.argmax(power))
            segments.append((float(start_sec), float(end_sec), f"SPEAKER_{channel + 1:02d}"))
        return segments
