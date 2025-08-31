import warnings
from pathlib import Path
from typing import List, Union
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset
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

    def __getitem__(self, idx):
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
            "labels": tokenized_transcript,
        }


if __name__ == "__main__":
    data = LibriSpeechDataset("/home/a/LibriSpeech")
    print(len(data))
    d = data[0]
    print(d)
    print(d["input_values"].shape)
