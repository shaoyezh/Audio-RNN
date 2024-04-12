import argparse

from sklearn.metrics import accuracy_score
from data_utils import *
from dataset import construct_datasets, zscore_normalize, minmax_normalize
from model import SimpleAudioRnn, ComplexAudioRnn, AudioRnn, ComplexAudioRnn_2
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
from typing import Union
import random
import matplotlib.pyplot as plt
from collections import OrderedDict


name_to_opt = {
    "adam": optim.Adam,
    "sgd": optim.SGD
}


def parse_params(args: argparse.Namespace) -> Tuple[int, str, float, int]:
    """
    Return the parameters given in necessary order.
    """
    return (
        args.batch_size,
        args.optimizer,
        args.lr,
        args.epochs,
    )


def run_one_iteration(model: AudioRnn, inputs: Tensor, inputs_lengths: Tensor, labels: Tensor,
                      optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.StepLR, args: argparse.Namespace,
                      criterion: nn.CrossEntropyLoss, mode: str="train") -> Tuple[Tensor, Tensor]:
    """
    Run 1 iteration of the given batch through the model.
    """
    inputs = inputs.float().to(args.device)
    inputs_lengths = inputs_lengths.to(args.device)
    labels = labels.to(args.device)
    logits = model(inputs, inputs_lengths.to(args.device))
    loss = criterion(logits, labels.to(args.device))

    if mode == "train":
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        scheduler.step()

    return logits, loss


def get_accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    Compute the accuracy of the batch given logits and labels.
    """
    predictions = logits.softmax(dim=1).argmax(dim=1)
    # print(torch.cat([predictions.unsqueeze(0), labels.unsqueeze(0)]).transpose(0, 1))
    correct = torch.eq(predictions, labels).sum()

    return correct.item() / labels.size(0)


def run_through_data(model: AudioRnn, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.StepLR,
                     criterion: nn.CrossEntropyLoss, data_loader: DataLoader,
                     args: argparse.Namespace, mode: str="train") -> Union[None, Tuple[float, float]]:
    """
    Run the model through the entire dataset. Mode must be one of train or validate
    """
    assert mode in ["train", "validate"], "<mode> argument must be one of \"train\" or \"validate\"."
    if mode == "train":
        model.train()
    else:
        model.eval()

    losses = []
    accuracies = []

    iteration = 1
    for inputs, inputs_lengths, labels in data_loader:
        inputs = minmax_normalize(inputs)
        inputs = inputs.permute(1, 0, 2).contiguous()
        logits, loss = run_one_iteration(
            model, inputs, inputs_lengths, labels, optimizer, scheduler, args, criterion, mode)
        accuracy = get_accuracy(logits, labels.to(args.device))

        if mode == "validate":
            accuracies.append(accuracy)
            losses.append(loss.item())
        else:
            print(f"Iteration: {iteration}. Training Loss: {loss.item():.3f}. Training Accuracy: {accuracy:.3f}")

        iteration += 1

    if mode == "validate":
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def train(model: AudioRnn, args: argparse.Namespace, train_dataloader: DataLoader,
          val_dataloader: DataLoader) -> None:
    """
    The training loop for the model. The criterion used will be Cross Entropy loss.
    """
    batch_size, opt_name, lr, epochs = parse_params(args)

    optimizer = name_to_opt[opt_name](model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, (len(train_dataloader) * args.epochs) // 4)
    criterion = nn.CrossEntropyLoss(torch.Tensor([2, 1]).to(args.device))

    for epoch in range(epochs):
        print(f"Training epoch {epoch + 1}")
        run_through_data(model, optimizer, scheduler, criterion, train_dataloader, args, "train")
        loss, accuracy = run_through_data(model, optimizer, scheduler, criterion, val_dataloader, args, "validate")
        print(f"End of epoch {epoch + 1}. Validation Loss: {loss:.3f}. Validation Accuracy: {accuracy:.3f}")


def main(args: argparse.Namespace) -> None:
    """
    Run the training using the given arguments.
    """
    model = ComplexAudioRnn(args.input_size, args.hidden_size)

    for name, _ in model.named_parameters():
        if name.startswith("bias"):
            bias = getattr(model, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(-1.)

    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    model = model.to(args.device)
    inputs, labels = pool_data(args.source)
    train_dataset, val_dataset, test_dataset = construct_datasets(
        inputs, labels, args.train_split, args.val_split)

    train_dataloader = DataLoader(
        train_dataset, args.batch_size, sampler=SequentialSampler(train_dataset), collate_fn=pad_and_sort_batch)
    val_dataloader = DataLoader(
        val_dataset, args.batch_size, sampler=SequentialSampler(val_dataset), collate_fn=pad_and_sort_batch)
    test_dataloader = DataLoader(
        test_dataset, args.batch_size, sampler=SequentialSampler(test_dataset), collate_fn=pad_and_sort_batch)

    train(model, args, train_dataloader, val_dataloader)

    accuracies = torch.empty(len(test_dataloader))
    for i, (inputs, inputs_lengths, labels) in enumerate(test_dataloader):
        inputs = minmax_normalize(inputs)
        inputs = inputs.permute(1, 0, 2).contiguous()
        logits = model(inputs.float().to(args.device), inputs_lengths.to(args.device))
        accuracies[i] = get_accuracy(logits, labels.to(args.device))

    print(f"Model test accuracy: {accuracies.mean():.3f}")


def test_decoder_seeds(args: argparse.Namespace) -> None:
    """
    Run the training using the given arguments.
    """
    #seeds = np.random.randint(low=0,high=100,size=5)
    seeds = [111, 34 ,57, 25]
    model1_test_accuracy = []
    model2_test_accuracy = []
    for seed in seeds:
        args.seed = seed
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        print(seed)
        model1 = ComplexAudioRnn(args.input_size, args.hidden_size)
        a1 = get_test_accuracy(args, model1)
        model1_test_accuracy.append(a1)
        model2 = ComplexAudioRnn_2(args.input_size, args.hidden_size)
        a2 = get_test_accuracy(args, model2)
        model2_test_accuracy.append(a2)
        print(a1, a2)
    plot_bar_chart(model1_test_accuracy, model2_test_accuracy, seeds, args)
        

def plot_bar_chart(data_1, data_2, seeds, args):
    x = np.arange(4)
    y1 = data_1
    y2 = data_2
    width = 0.2
    
    # plot data in grouped manner of bar type
    plt.bar(x-0.2, y1, width, color='cyan')
    plt.bar(x, y2, width, color='orange')
    plt.xticks(x, seeds)
    plt.xlabel("Seeds")
    plt.ylabel("Test Accuracy")
    plt.legend(["1 layer decoder", "2 layer decoder"])
    plt.title("h:{}, lr:{},b:{},e:{}".format(args.hidden_size, args.lr, args.batch_size, args.epochs))
    plt.show()

def get_test_accuracy(args: argparse.Namespace, model:AudioRnn):
    for name, _ in model.named_parameters():
        if name.startswith("bias"):
            bias = getattr(model, name)
            n = bias.size(0)
            start, end = n // 4, n // 2
            bias.data[start:end].fill_(-1.)

    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    model = model.to(args.device)
    inputs, labels = pool_data(args.source)
    train_dataset, val_dataset, test_dataset = construct_datasets(
        inputs, labels, 0.8, 0.1)

    train_dataloader = DataLoader(
        train_dataset, args.batch_size, sampler=SequentialSampler(train_dataset), collate_fn=pad_and_sort_batch)
    val_dataloader = DataLoader(
        val_dataset, args.batch_size, sampler=SequentialSampler(val_dataset), collate_fn=pad_and_sort_batch)
    test_dataloader = DataLoader(
        test_dataset, args.batch_size, sampler=SequentialSampler(test_dataset), collate_fn=pad_and_sort_batch)

    train(model, args, train_dataloader, val_dataloader)

    accuracies = torch.empty(len(test_dataloader))
    for i, (inputs, inputs_lengths, labels) in enumerate(test_dataloader):
        inputs = minmax_normalize(inputs)
        inputs = inputs.permute(1, 0, 2).contiguous()
        logits = model(inputs.float().to(args.device), inputs_lengths.to(args.device))
        accuracies[i] = get_accuracy(logits, labels.to(args.device))

    return accuracies.mean()


def test_hidden_size(args: argparse.Namespace):
    accuracies = []
    hiddens = [5,6,7,8,9,10,11,12,13,14,15]
    for h in hiddens:
        args.hidden_size = h 
        model1 = ComplexAudioRnn(args.input_size, args.hidden_size)
        a1 = get_test_accuracy(args, model1)
        accuracies.append(a1)
    
    plt.plot(hiddens, accuracies)
    plt.xlabel("hidden size")
    plt.ylabel("test accuracy")
    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True, help="Path to directory containing subdirectories of data", type=str)
    parser.add_argument("--input_size", help="Dimensionality of data", type=int, default=13)
    parser.add_argument("--hidden_size", help="Number of hidden neurons to use", type=int, default=16)
    parser.add_argument("--batch_size", help="Batch size to use for training", type=int, default=32)
    parser.add_argument("--optimizer", help="Name of optimizer to use for training, one of adam or sgd", type=str,
                        choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--lr", help="Learning rate to use for training", type=float, default=1e-4)
    parser.add_argument("--epochs", help="The number of epochs to run training for", type=int, default=8),
    parser.add_argument("--train_split", help="The percentage of the data used for training", type=float, default=0.7)
    parser.add_argument("--val_split", help="The percentage of the data use for validation", type=float, default=0.2)
    parser.add_argument("--no_cuda", help="Disable CUDA training", action="store_true")
    parser.add_argument("--seed", help="Number for seeding random modules", type=int, default=42)

    arguments = parser.parse_args()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    #test_decoder_seeds(arguments)
    test_hidden_size(arguments)
