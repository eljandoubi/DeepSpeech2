import warnings
from pathlib import Path
from typing import List, Union, Dict

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2CTCTokenizer

warnings.filterwarnings("ignore")


class LibriSpeechDataset(Dataset):
    """
    LibriSpeechDataset downloaded from OpenSLR: https://www.openslr.org/12

    There are 5 splits downloaded, 3 which are for training and 3 for testing:

        Training: ["train-clean-100", "train-clean-360", "train-other-500"]
        Validation: ["dev-clean", "test-clean"]

    """

    def __init__(
        self,
        path_to_data_root: str,
        include_splits: Union[str, List[str]] = "dev-clean",
        sampling_rate: int = 16000,
        num_audio_channels: int = 1,
    ):
        if isinstance(include_splits, str):
            include_splits = [include_splits]

        self.root = Path(path_to_data_root)
        assert self.root.is_dir(), f"{self.root} must be a directory"
        self.sampling_rate = sampling_rate
        self.num_audio_channels = num_audio_channels

        ### GET PATH TO ALL AUDIO/TEXT FILES ###
        self.librispeech_data = []
        for split in include_splits:
            path_to_split = self.root / split
            for trans_file in path_to_split.glob("**/*.txt"):
                transcripts = trans_file.read_text().strip().split("\n")
                for line in transcripts:
                    split_line = line.split()
                    full_path_to_audio_file = trans_file.parent / (
                        split_line.pop(0) + ".flac"
                    )
                    transcript = " ".join(split_line).strip()
                    self.librispeech_data.append((full_path_to_audio_file, transcript))

        self.audio2mels = T.MelSpectrogram(sample_rate=sampling_rate, n_mels=80)

        self.amp2db = T.AmplitudeToDB(top_db=80.0)

        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

    def __len__(self):
        return len(self.librispeech_data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ### Grab Path to Audio and Transcript ###
        path_to_audio, transcript = self.librispeech_data[idx]

        ### Load Audio ###
        audio, orig_sr = torchaudio.load(path_to_audio)

        if orig_sr != self.sampling_rate:
            audio = F.resample(audio, orig_freq=orig_sr, new_freq=self.sampling_rate)

        ### Create Mel Spectrogram and Convert it to Decibels ###
        mel = self.amp2db(self.audio2mels(audio))

        ### Normalize Spectrogram ###
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        ### Tokenize Text ###
        tokenized_transcript = self.tokenizer.encode(transcript, return_tensors="pt")

        return {
            "input_values": mel.squeeze(0).transpose(0, 1),
            "labels": tokenized_transcript.squeeze(0),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    This collate function is basically the heart of our implementation! It includes everything we need for training
    such as attention masks, sub_attention_masks, span_masks and our sampled negatives!
    """

    ### Sort Batch from Longest to Shortest (for future packed padding) ###
    batch = sorted(batch, key=lambda x: x["input_values"].shape[0], reverse=True)

    ### Grab Audios from our Batch Dictionary ###
    batch_mels = [sample["input_values"] for sample in batch]
    batch_transcripts = [sample["labels"] for sample in batch]

    ### Get Length of Audios ###
    seq_lens = torch.tensor([b.shape[0] for b in batch_mels], dtype=torch.long)

    ### Pad and Stack Spectrograms ###
    spectrograms = torch.nn.utils.rnn.pad_sequence(
        batch_mels, batch_first=True, padding_value=0
    )

    ### Convert to Shape Convolution Is Happy With (B x C x H x W) ###
    spectrograms = spectrograms.unsqueeze(1).transpose(-1, -2)

    ### Get Target Lengths ###
    target_lengths = torch.tensor(
        [t.shape[0] for t in batch_transcripts], dtype=torch.long
    )

    ### Pack Transcripts (CTC Loss Can Take Packed Targets) ###
    packed_transcripts = torch.cat(batch_transcripts)

    return {
        "input_values": spectrograms,
        "seq_lens": seq_lens,
        "labels": packed_transcripts,
        "target_lengths": target_lengths,
    }


if __name__ == "__main__":
    dataset = LibriSpeechDataset("/home/a/LibriSpeech")
    ### Test Collate Function ###
    loader = DataLoader(dataset, batch_size=5, collate_fn=collate_fn)
    batch = next(iter(loader))

    print("Input Values:", batch["input_values"].shape)
    print("Seq Lens", batch["seq_lens"])
    print("Labels:", batch["labels"].shape)
    print("Target Lengths:", batch["target_lengths"])

    ### As required by the CTC loss, sum of the target lengths must equal the length of the flattened labels ###
    if batch["target_lengths"].sum() == len(batch["labels"]):
        print("Sucess, Same Length")
