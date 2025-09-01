import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup, HfArgumentParser

from model import DeepSpeech2
from config import TrainConfig
from data import DataLoader, LibriSpeechDataset, collate_fn, Wav2Vec2CTCTokenizer


def training_loop(cfg: TrainConfig):
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")
    trainset = LibriSpeechDataset(
        path_to_data_root=cfg.dataset_root,
        tokenizer=tokenizer,
        include_splits=["train-clean-100", "train-clean-360", "train-other-500"],
    )
    valset = LibriSpeechDataset(
        path_to_data_root=cfg.dataset_root,
        tokenizer=tokenizer,
        include_splits=["dev-clean", "dev-other"],
    )
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory="cuda" in cfg.device,
    )
    valloader = DataLoader(
        valset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory="cuda" in cfg.device,
    )

    ### DEFINE MODEL ###
    model = DeepSpeech2(
        conv_in_channels=1, conv_out_channels=32, rnn_hidden_size=512
    ).to(cfg.device)

    params = sum([p.numel() for p in model.parameters()])
    print("Total Training Parameters:", params)

    ### OPTIMIZER/SCHEDULER ###
    optimizer = optim.AdamW(params=model.parameters(), lr=cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=cfg.training_iterations
    )

    ### HOUSEKEEPING ###
    best_val_loss = np.inf
    train = True
    completed_steps = 0
    train_loss, validation_loss = [], []
    pbar = tqdm(range(cfg.training_iterations))

    ### TRAINING LOOP ###
    while train:
        training_losses = []
        validation_losses = []

        for batch in trainloader:
            ### Pass Through Model and get input_lengths (post convolutions) and logits ###
            logits, input_lengths = model(
                x=batch["input_values"].to(cfg.device), seq_lens=batch["seq_lens"]
            )

            ### CTC Expects Log Probabilities ###
            log_probs = nn.functional.log_softmax(logits, dim=-1)

            ### CTC Also Expects (T x B x C), we have (B x T x C) ###
            log_probs = log_probs.transpose(0, 1)

            ### Compute CTC Loss ###
            loss = nn.functional.ctc_loss(
                log_probs=log_probs,
                targets=batch["labels"].to(cfg.device),
                input_lengths=input_lengths,
                target_lengths=batch["target_lengths"],
                blank=tokenizer.pad_token_id,
                reduction="mean",
            )

            ### Update Model ###
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            ### Store Loss ###
            training_losses.append(loss.item())

            ### Iterate Counter ###
            completed_steps += 1
            pbar.update(1)

            if completed_steps % cfg.eval_iterations == 0:
                print("Evaluating")

                model.eval()

                for batch in tqdm(valloader):
                    ### Pass Through Model and get input_lengths (post convolutions) and logits ###
                    with torch.no_grad():
                        logits, input_lengths = model(
                            x=batch["input_values"].to(cfg.device),
                            seq_lens=batch["seq_lens"],
                        )

                    ### CTC Expects Log Probabilities ###
                    log_probs = nn.functional.log_softmax(logits, dim=-1)

                    ### CTC Also Expects (T x B x C), we have (B x T x C) ###
                    log_probs = log_probs.transpose(0, 1)

                    ### Compute CTC Loss ###
                    loss = nn.functional.ctc_loss(
                        log_probs=log_probs,
                        targets=batch["labels"].to(cfg.device),
                        input_lengths=input_lengths,
                        target_lengths=batch["target_lengths"],
                        blank=tokenizer.pad_token_id,
                        reduction="mean",
                    )

                    ### Store Loss ###
                    validation_losses.append(loss.item())

                training_loss_mean = np.mean(training_losses)
                valid_loss_mean = np.mean(validation_losses)

                train_loss.append(training_loss_mean)
                validation_loss.append(valid_loss_mean)

                ### Save Model If Val Loss Decreases ###
                if valid_loss_mean < best_val_loss:
                    print("---Saving Model---")
                    torch.save(model.state_dict(), "best_weights.pt")
                    best_val_loss = valid_loss_mean

                print("Training Loss:", training_loss_mean)
                print("Validation Loss:", valid_loss_mean)

                ### Reset Lists for Store ###
                training_losses = []
                validation_losses = []

                ### Set Model to Training Mode ###
                model.train()

            if completed_steps >= cfg.training_iterations:
                train = False
                print("Completed Training")
                break

    return train_loss, validation_loss


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = HfArgumentParser(TrainConfig).parse_args_into_dataclasses()[0]
    train_loss, validation_loss = training_loop(config)
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, validation_loss, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("Losses.png")
