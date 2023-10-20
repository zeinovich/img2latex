import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from torch.utils.data import DataLoader

from ..data import IM2LaTEX100K  # Dataset
from ..data.utils import read_input
from ..models.base_model import BaseIm2SeqModel
from .utils import edit_distance, make_training_report

import os
import importlib
import argparse
from tqdm import tqdm
from datetime import datetime


# [TODO] ABC for model
def import_class(name: str) -> BaseIm2SeqModel:
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trains multimodal model on LaTEX Dataset"
    )
    parser.add_argument(
        "-m",
        "--model-arch",
        type=str,
        required=True,
        help="Model architecture to be trained. \
Only img2latex.models.cnn_lstm.CNNLSTM available",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON model config. \
If not specified, default model initialized by default",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="NA",
        help="Model checkpoint folder path. Saves model checkpoint after 10 epochs. \
If not specified saves to ./checkpoints/model.pt",
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        type=int,
        default=50,
        help="Number of epochs of training. Defaults to 50",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. If not specified equals to 1 with CPU and 8 with GPU",
    )
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        nargs="+",
        help="LaTEX datasets. Must be 'train', 'val', 'test'",
    )
    parser.add_argument(
        "--image-folder", type=str, required=True, help="Image folder path"
    )
    parser.add_argument(
        "--vocab", type=str, required=True, help="Vocabulary JSON file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device training will be performed on",
    )

    return parser.parse_args()


def training_func(
    model: BaseIm2SeqModel,
    n_epochs: int,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    dataloaders: dict[str, DataLoader],
    checkpoint: str,
    end_token: int,
) -> None:
    # training consists of 3 phases: train, val, test
    # test is performed only once after model has been
    # trained for NUM_EPOCHS

    lengths = {phase: len(dl) for phase, dl in dataloaders.items()}
    history = {}

    # for checkpoints
    keep = 2  # number of checkpoints to keep
    _last_saved = []  # cached paths of checkpoints
    # range (1, n_epochs + 1) for prettier monitoring
    for epoch in range(1, n_epochs + 1):
        losses = {"train": 0, "val": 0, "test": 0}
        num_correct = {"train": 0, "val": 0, "test": 0}
        num_total = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        total_distances = {"train": 0, "val": 0, "test": 0}
        levenstein = {"train": 0, "val": 0, "test": 0}
        min_loss = float("inf")

        for phase in ["train", "val", "test"]:
            if phase == "train":
                model.train()
            else:
                # eval on val and test
                model.eval()

            if phase == "test" and epoch != n_epochs:
                # skip test phase if not last epoch
                continue

            # tqdm progress bar
            with tqdm(
                dataloaders[phase], miniters=1, unit="batch"
            ) as pbar, torch.set_grad_enabled(phase == "train"):
                # if phase != "train" works as torch.no_grad()
                for img, tokens in pbar:
                    pbar.set_description_str(
                        f"{phase.capitalize():5}({epoch:03d})"
                    )

                    # tokens [batch_size, max_len]
                    # output [batch_size, max_len, len(vocab)]
                    img = img.to(model.device)
                    tokens = tokens.to(model.device)
                    output = model(img)

                    # reshape target and output for loss
                    # output_flat [(max_len - 1) * batch_size, len(vocab)]
                    # target_flat [(max_len - 1) * batch_size]
                    output_dim = output.shape[-1]
                    output_flat = output[:, 1:].view(-1, output_dim)
                    tokens_flat = tokens[:, 1:].view(-1)

                    loss = loss_fn(output_flat, tokens_flat)

                    # will raise error with no grads
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    # scale loss by length of dataloader to obtain avg loss
                    losses[phase] += loss.item() / lengths[phase]

                    # [TODO] Mb ignore padding ???
                    predictions = output.argmax(2)

                    # get index of <EOS> token guaranteed to be only one in tokens
                    end_token_idx = (tokens == end_token).nonzero(
                        as_tuple=True
                    )[-1]
                    # truncate to only valuable output
                    # after hitting EOS tokens all true tokens will be <PAD>
                    predictions = predictions[:, 1:end_token_idx]
                    tokens = tokens[:, 1:end_token_idx]

                    # compute accuracy
                    num_correct[phase] += torch.sum(
                        predictions == tokens,
                        dim=1,
                    ).item()

                    num_total[phase] += tokens.size(0) * tokens.size(1)
                    accuracies[phase] = num_correct[phase] / num_total[phase]

                    # compute edit distances over batch
                    for b in range(tokens.size(0)):
                        total_distances[phase] += edit_distance(
                            predictions[b], tokens[b]
                        )
                    # average over epoch
                    levenstein[phase] = (
                        total_distances[phase]
                        # number of sequences processed = total_tokens / seq_len
                        / (num_total[phase] / tokens.size(1))
                    )

                    # make progress string
                    report = make_training_report(
                        losses[phase],
                        levenstein[phase],
                        accuracies[phase],
                        scheduler.get_last_lr()[0],
                    )
                    pbar.set_postfix_str(report)

                # step scheduler after epoch if training
                if phase == "train":
                    lr = scheduler.get_last_lr()[0]
                    scheduler.step()

            # save training history
            history[epoch] = {
                "acc": accuracies,
                "edit_dist": levenstein,
                "loss": losses,
                "lr": lr,
            }

            if losses["val"] < min_loss:
                min_loss = losses["val"]
                file_path = f"{checkpoint}/{datetime.now().strftime('%d%m-%H%M%S')}-acc-{accuracies['val']:.3f}.pth"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "history": history,
                    },
                    file_path,
                )

                _last_saved.append(file_path)

                # remove file if more than KEEP is stored
                if len(_last_saved) > keep:
                    oldest = _last_saved.pop(0)

                    if os.path.exists(oldest):
                        os.remove(oldest)

    print(f"Last checkpoint: {_last_saved[-1]}")
    return history, model


def train_model(
    model_arch: str,
    checkpoint: str,
    data: str,
    batch_size: str,
    num_epochs: int,
    device: torch.device,
    vocab: str,
    image_folder: str,
):
    vocab = read_input(vocab)

    model_class = import_class(model_arch)
    # [TODO] do not pass vocab only pass vocab length

    model = model_class(vocab_len=len(vocab), device=device)

    if checkpoint == "NA":
        checkpoint = f"./artifacts/{model_class.__name__}/"

    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)

    # if batch_size is not specified
    if batch_size == 0:
        batch_size = 1 if device == "cpu" else 8

    # [TODO] Use model.input_size member
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Grayscale(),
        ]
    )

    phases = ["train", "val", "test"]
    datasets = {
        phase: IM2LaTEX100K(
            file=file,
            transform=transforms,
            vocab_len=len(vocab),
            image_folder=image_folder,
        )
        for phase, file in zip(phases, data)
    }
    dataloaders = {
        phase: DataLoader(dataset, batch_size=batch_size)
        for phase, dataset in datasets.items()
    }

    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = optim.Adam(params=model.parameters(), lr=5e-2)
    scheduler = CosineAnnealingLR(
        optimizer=optimizer, T_max=10, eta_min=1e-3, last_epoch=-1
    )

    model.to(device)

    training_func(
        model=model,
        n_epochs=num_epochs,
        loss_fn=ce_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        checkpoint=checkpoint,
        end_token=vocab["<EOS>"],
    )

    return


def main():
    args = vars(cli())
    args = {
        arg.replace("-", "_"): v for arg, v in args.items() if v is not None
    }
    print(args)

    train_model(**args)


if __name__ == "__main__":
    main()

# python -m img2latex.training.train -m img2latex.models.cnn_lstm.ResnetLSTM -d ./data/interim/tokens.debug.csv ./data/interim/tokens.debug.csv ./data/interim/tokens.debug.csv --vocab ./data/processed/token2id.json --image-folder data/raw/formula_images_processed/formula_images_processed/ --num-epochs 200
