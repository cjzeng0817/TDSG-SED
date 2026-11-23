#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
from torch.utils.data import Dataset
import torchaudio

import numpy as np
import pandas as pd
import os.path
from glob import glob
from pathlib import Path
import wave
try:
    import soundfile as sf
except ImportError:
    sf = None


#dataset classes
class StronglyLabeledDataset(Dataset):
    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        #refer: https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * encoder.sr
        self.return_name = return_name

        #construct clip dictionary with filename = {path, events} where events = {label, onset and offset}
        clips = {}
        for _, row in tsv_read.iterrows():
            if row["filename"] not in clips.keys():
                clips[row["filename"]] = {"path": os.path.join(dataset_dir, row["filename"]), "events": []}
            if not np.isnan(row["onset"]):
                clips[row["filename"]]["events"].append({"event_label": row["event_label"],
                                                         "onset": row["onset"],
                                                         "offset": row["offset"]})
        self.clips = clips #dictionary for each clip
        self.clip_list = list(clips.keys()) # list of all clip names

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)

        # get labels
        events = clip["events"]
        if not len(events): #label size = [frames, nclass]
            label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels)).float()
        else:
            label = self.encoder.encode_strong_df(pd.DataFrame(events))
            label = torch.from_numpy(label).float()
        label = label.transpose(0, 1)

        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


class WeaklyLabeledDataset(Dataset):
    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * self.encoder.sr
        self.return_name = return_name

        #construct clip dictionary with file name, path, label, onset and offset
        clips = {}
        for _, row in tsv_read.iterrows():
            if row["filename"] not in clips.keys():
                clips[row["filename"]] = {"path": os.path.join(dataset_dir, row["filename"]),
                                          "events": row["event_labels"].split(",")}
        #dictionary for each clip
        self.clips = clips
        self.clip_list = list(clips.keys()) # all file names

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        #get labels
        events = clip["events"]
        label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels))
        if len(events):
            label_encoded = self.encoder.encode_weak(events)      # label size: [n_class]
            label[0, :] = torch.from_numpy(label_encoded).float() # label size: [n_frames, n_class]
        label = label.transpose(0, 1)

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)
        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


class UnlabeledDataset(Dataset):
    def __init__(self, dataset_dir, return_name, encoder):
        self.encoder = encoder
        self.pad_to = encoder.audio_len * self.encoder.sr
        self.return_name = return_name

        #list of clip directories
        self.clips = glob(os.path.join(dataset_dir, '*.wav'))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.clips[idx]
        filename = os.path.split(path)[-1]

        #produce empty label
        label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels)).float()
        label = label.transpose(0, 1)

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)
        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


def waveform_modification(filepath, pad_to, encoder):
    if sf is not None:
        wav, sr = sf.read(filepath)
    else:
        wav_tensor, sr = torchaudio.load(filepath)
        wav = wav_tensor.transpose(0, 1).numpy()
    if sr != encoder.sr:
        raise ValueError(
            "Expected %d Hz audio but got %d Hz for %s. "
            "Run scripts/resample_urban_sed_16k.py before training."
            % (encoder.sr, sr, filepath)
        )
    wav = to_mono(wav)
    wav, pad_mask = pad_wav(wav, pad_to, encoder)
    wav = torch.from_numpy(wav).float()
    wav = normalize_wav(wav)
    return wav, pad_mask


def normalize_wav(wav):
    return wav / (torch.max(torch.max(wav), -torch.min(wav)) + 1e-10)


def to_mono(wav, rand_ch=False):
    if wav.ndim > 1:
        if rand_ch:
            ch_idx = np.random.randint(0, wav.shape[-1] - 1)
            wav = wav[:, ch_idx]
        else:
            wav = np.mean(wav, axis=-1)
    return wav


def pad_wav(wav, pad_to, encoder):
    if len(wav) < pad_to:
        pad_from = len(wav)
        wav = np.pad(wav, (0, pad_to - len(wav)), mode="constant")
    else:
        wav = wav[:pad_to]
        pad_from = pad_to
    pad_idx = np.ceil(encoder._time_to_frame(pad_from / encoder.sr))
    pad_mask = torch.arange(encoder.n_frames) >= pad_idx # size = n_frame, [0, 0, 0, 0, 0, ..., 0, 1, ..., 1]
    return wav, pad_mask


def setmelspectrogram(feature_cfg):
    return torchaudio.transforms.MelSpectrogram(sample_rate=feature_cfg["sample_rate"],
                                                n_fft=feature_cfg["n_window"],
                                                win_length=feature_cfg["n_window"],
                                                hop_length=feature_cfg["hop_length"],
                                                f_min=feature_cfg["f_min"],
                                                f_max=feature_cfg["f_max"],
                                                n_mels=feature_cfg["n_mels"],
                                                window_fn=torch.hamming_window,
                                                wkwargs={"periodic": False},
                                                power=1) # 1:energy, 2:power


def get_audio_duration(filepath):
    try:
        info = sf.info(str(filepath))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        with wave.open(str(filepath), "rb") as wav_file:
            return float(wav_file.getnframes()) / float(wav_file.getframerate())


def merge_overlapping_same_class_events(events_df):
    if events_df.empty:
        return events_df

    empty_events = events_df[events_df["event_label"].isna()].copy()
    events = events_df.dropna(subset=["event_label", "onset", "offset"]).copy()
    if events.empty:
        return events_df

    merged_rows = []
    for (filename, event_label), group in events.groupby(["filename", "event_label"], sort=False):
        group = group.sort_values(["onset", "offset"])
        current_onset = None
        current_offset = None
        for _, row in group.iterrows():
            onset = float(row["onset"])
            offset = float(row["offset"])
            if current_onset is None:
                current_onset = onset
                current_offset = offset
            elif onset <= current_offset:
                current_offset = max(current_offset, offset)
            else:
                merged_rows.append({
                    "filename": filename,
                    "onset": current_onset,
                    "offset": current_offset,
                    "event_label": event_label,
                })
                current_onset = onset
                current_offset = offset

        merged_rows.append({
            "filename": filename,
            "onset": current_onset,
            "offset": current_offset,
            "event_label": event_label,
        })

    merged = pd.DataFrame(merged_rows, columns=["filename", "onset", "offset", "event_label"])
    if not empty_events.empty:
        merged = pd.concat([merged, empty_events], ignore_index=True)
    return merged.sort_values(["filename", "onset", "offset", "event_label"]).reset_index(drop=True)


def read_urban_sed_split(annotation_dir, audio_dir):
    annotation_dir = Path(annotation_dir)
    audio_dir = Path(audio_dir)
    if not annotation_dir.exists():
        raise FileNotFoundError(
            "Missing URBAN-SED annotation directory: %s. "
            "Run scripts/resample_urban_sed_16k.py before training if using the _16k config."
            % annotation_dir
        )
    if not audio_dir.exists():
        raise FileNotFoundError(
            "Missing URBAN-SED audio directory: %s. "
            "Run scripts/resample_urban_sed_16k.py before training."
            % audio_dir
        )

    rows = []
    durations = []
    annotation_paths = sorted(annotation_dir.glob("*.txt"))
    if not annotation_paths:
        raise ValueError("No URBAN-SED .txt annotations found in %s" % annotation_dir)

    for annotation_path in annotation_paths:
        filename = annotation_path.with_suffix(".wav").name
        audio_path = audio_dir / filename
        if not audio_path.exists():
            raise FileNotFoundError("Missing audio file for annotation: %s" % audio_path)

        duration = get_audio_duration(audio_path)
        durations.append({"filename": filename, "duration": duration})
        has_events = False
        with annotation_path.open("r") as annotation_file:
            for line_number, line in enumerate(annotation_file, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    parts = line.split()
                if len(parts) != 3:
                    raise ValueError(
                        "Invalid URBAN-SED annotation line %s:%d: %s"
                        % (annotation_path, line_number, line)
                    )
                onset, offset, event_label = parts
                onset = min(max(float(onset), 0.0), duration)
                offset = min(max(float(offset), onset), duration)
                rows.append({
                    "filename": filename,
                    "onset": onset,
                    "offset": offset,
                    "event_label": event_label,
                })
                has_events = True

        if not has_events:
            rows.append({
                "filename": filename,
                "onset": np.nan,
                "offset": np.nan,
                "event_label": np.nan,
            })

    annotations = pd.DataFrame(rows, columns=["filename", "onset", "offset", "event_label"])
    annotations = merge_overlapping_same_class_events(annotations)
    durations = pd.DataFrame(durations, columns=["filename", "duration"])
    return annotations, durations


def prepare_urban_sed_metadata(dataset_cfg):
    root_dir = Path(dataset_cfg.get("root_dir", "URBAN-SED_v2.0.0"))
    metadata_dir = Path(dataset_cfg.get("metadata_folder", root_dir / "metadata"))
    metadata_dir.mkdir(parents=True, exist_ok=True)

    split_defaults = {
        "train": ("train", "train"),
        "valid": ("validate", "validate"),
        "test": ("test", "test"),
    }
    metadata = {}
    for split_name, (audio_split, annotation_split) in split_defaults.items():
        audio_dir = Path(dataset_cfg.get("%s_folder" % split_name, root_dir / "audio" / audio_split))
        annotation_dir = Path(dataset_cfg.get("%s_annotations" % split_name, root_dir / "annotations" / annotation_split))
        annotations, durations = read_urban_sed_split(annotation_dir, audio_dir)

        tsv_path = metadata_dir / ("%s.tsv" % split_name)
        durations_path = metadata_dir / ("%s_durations.tsv" % split_name)
        annotations.to_csv(tsv_path, sep="\t", index=False)
        durations.to_csv(durations_path, sep="\t", index=False)

        metadata[split_name] = {
            "audio_dir": str(audio_dir),
            "tsv": str(tsv_path),
            "durations": str(durations_path),
            "annotations": annotations,
            "durations_df": durations,
        }

    return metadata
