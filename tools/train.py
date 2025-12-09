from utils.tokenize import token_ids_to_text, text_to_token_ids, generate_text_simple
from utils.loss import calc_loss_batch, calc_loss_loader
from utils.text_processing import read_config, read_file
from dataset.dataset import create_dataloader_v1
from model.gpt_model import GPTModel
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse
import torch
import sys
import os

def arg_parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/config.yaml")
    parser.add_argument("--model", type=str, default="GPT_CONFIG_124M")
    return parser.parse_args(argv)

def train(args):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load config
    config = read_config(args.config_path)
    GPT_CONFIG = config[args.model]
    HYPERPARAMS = config["hyperparams"]
    data_config = config["data"]

    # load text data
    text_data = read_file(data_config["textfile_path"])
    train_data = text_data[:int(data_config["train_ratio"] * len(text_data))]
    val_data = text_data[int(data_config["train_ratio"] * len(text_data)):]
    
    # set tokenizer
    tokenizer = tiktoken.get_encoding(data_config['tokenizer'])

    # make dataloader
    train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG['context_length'],
    stride=GPT_CONFIG['context_length'],
    shuffle=True,
    drop_last=True,
    num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG['context_length'],
        stride=GPT_CONFIG['context_length'],
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    # load model
    model = GPTModel(GPT_CONFIG)
    model.to(device)
    # load optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    # set variables for train
    num_epochs = HYPERPARAMS["num_epochs"]
    eval_freq = HYPERPARAMS["eval_freq"]
    eval_iter = HYPERPARAMS["eval_iter"]
    start_context = HYPERPARAMS["start_context"]

    # set manual seed for reproduction
    torch.manual_seed(123)

    train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
    start_context=start_context, tokenizer=tokenizer
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, data_config["plot_dir"])

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_dir):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    # save plot
    fig.savefig(f"{plot_dir}/losses.png")


def train_model_simple(model, train_loader, val_loader,
                        optimizer, device, num_epochs,
                        eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"    
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(
    model, tokenizer, device, start_context
    ):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

if __name__ == '__main__':
    args = arg_parse(sys.argv[1:])
    train(args)


