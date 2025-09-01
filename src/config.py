from typing import Literal
from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 32
    device: Literal["cuda", "cpu"] = "cuda"
    training_iterations: int = 50000
    eval_iterations: int = 2500
    learning_rate: float = 1e-4
    num_workers: int = 32
    dataset_root: str = "./data/LibriSpeech"
