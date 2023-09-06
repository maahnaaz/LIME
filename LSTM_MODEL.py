
!pip install --upgrade torchtext

pip install portalocker

# Imports
import numpy as np
import pandas as pd
import json
import random
import math
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import torch
from sklearn.metrics import mean_squared_error
from mlxtend.preprocessing import TransactionEncoder
import random
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Callable, Iterable
import torchtext
import torchtext.transforms
from torch.hub import load_state_dict_from_url
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

# read the data (Train and Test)
from torchtext.datasets import AG_NEWS
from torchtext.data.functional import to_map_style_dataset
import portalocker

train_datapipe, test_datapipe = AG_NEWS()
Train_dataset = to_map_style_dataset(train_datapipe)
#train_dataset = [Train_dataset[i] for i in range(3000)] # Choose a subsample of Train_dataset
train_dataset = Train_dataset
Test_dataset = to_map_style_dataset(test_datapipe)
#test_dataset = [Test_dataset[i] for i in range(300)] # Choose a subsample of Test_dataset
test_dataset = Test_dataset

# Model F
class TokenizerModule(torch.nn.Module):
    def __init__(self, tokenizer_name: str) -> None:
        super().__init__()
        self.name = tokenizer_name
        self.tokenizer_fn: Callable[
            [str], list[str]
        ] = torchtext.data.utils.get_tokenizer(tokenizer_name)

    def forward(self, x: str) -> list[str]:
        return self.tokenizer_fn(x)

def train(
    train_dataset: list,
    test_dataset: list,
    model: torch.nn.Module,
    tokenizer: torch.nn.Module,
    vocab: torch.nn.Module,
    batch_size: int,
    learning_rate: float = 1e-5,
    epochs: int = 10,
    seed: int = 0,
    val_proportion: float = 0.05,
    padding_idx: int = 1,
    bos_idx: int = 0,
    eos_idx: int = 2,
    max_seq_len: int = 256,
) -> None:

    global device

    model = model.to(device)
    tokenizer = tokenizer.to(device)
    vocab = vocab.to(device)
    def label_transform(x: int) -> int:
        return x - 1

    text_transform = torchtext.transforms.Sequential(
        tokenizer,
        torchtext.transforms.VocabTransform(vocab),
        torchtext.transforms.Truncate(max_seq_len - 2),
        torchtext.transforms.AddToken(token=bos_idx, begin=True),
        torchtext.transforms.AddToken(token=eos_idx, begin=False),
    )
    def collate_batch(
        batch: Iterable[tuple[int, str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label_list, text_list, lengths = [], [], []
        for _label, _text in batch:
            label_list.append(label_transform(_label))
            token_ids = text_transform(_text)
            text_list.append(token_ids)
            lengths.append(len(token_ids))

        label_tensor = torch.tensor(label_list, dtype=torch.int64, device=device)
        length_tensor = torch.tensor(lengths, dtype=torch.int64)
        padded_text = torchtext.functional.to_tensor(
            text_list, padding_value=padding_idx
        ).to(device)
        return label_tensor, padded_text, length_tensor



    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [1 - val_proportion, val_proportion],
        generator=torch.Generator().manual_seed(seed),
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_batch,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_batch,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    def evaluate(data_loader: DataLoader[tuple[int, str]]) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        batch_count = 0
        with torch.no_grad():
            for target, text, lengths in data_loader:
                # print(len(target), len(text), len(lengths))
                prediction = model(text, lengths, trainable=True)
                # print(prediction.argmax(1))
                # print(target)
                # print()
                total_loss += loss_fn(prediction, target).item()
                correct_predictions += (prediction.argmax(1) == target).sum().item()
                total_predictions += target.size(0)
                batch_count += 1
        # print('length of data_loader, batch_count, total_predictions')
        # print(len(data_loader))
        # print(batch_count, total_predictions)
        return total_loss / batch_count, correct_predictions / total_predictions



    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        batch_count = 0
        for target, text, offsets in train_data_loader:
            optim.zero_grad()
            try:
              prediction = model(text, offsets, trainable=True)
            except:
              print('except')
              prediction = model(text, offsets)
            loss = loss_fn(prediction, target)
            loss.backward()
            optim.step()

            total_loss += loss_fn(prediction, target).item()
            correct_predictions += (prediction.argmax(1) == target).sum().item()
            total_predictions += target.size(0)
            batch_count += 1

        avg_train_batch_loss = total_loss / batch_count
        train_acc = correct_predictions / total_predictions

        # evaluate model
        avg_val_batch_loss, val_acc = evaluate(val_data_loader)
        print(
            f"Epoch {epoch:3}: "
            f"train loss = {avg_train_batch_loss:6.4f} "
            f"train acc = {train_acc*100:5.2f}% | "
            f"val loss = {avg_val_batch_loss:6.4f} "
            f"val. accuracy {val_acc*100:5.2f}%"
        )

    print("Testing...")
    avg_test_batch_loss, test_acc = evaluate(test_data_loader)
    print(
        f"avg. test. batch loss = {avg_test_batch_loss:.4f}, "
        f"test. accuracy {test_acc*100:.2f}%"
    )

    Path("models").mkdir(exist_ok=True)
    torch.save(
        model.state_dict(),
        f"models/{model.name}_{tokenizer.name}_{epochs}e_{learning_rate}lr.pt",
    )

class LinearEmbeddingBagModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_class: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.name = (
            f"linear_embedding_bag_{vocab_size}v_{embedding_dim}emb_{num_class}nc"
        )
        self.embedding = torch.nn.EmbeddingBag(
            vocab_size, embedding_dim, sparse=False, padding_idx=padding_idx
        )
        self.fc = torch.nn.Linear(embedding_dim, num_class)

    def forward(
        self, padded_text: torch.Tensor, _lengths: torch.Tensor, embedded_input: bool = False,
    ) -> torch.Tensor:
        if embedded_input:
            text_embedding = padded_text
        else:
            text_embedding = self.embedding(padded_text)

        prediction: torch.Tensor = self.fc(text_embedding)
        return prediction

class SimpleBidirectionalLSTM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_class: int,
        padding_idx: int | None = 1,
        bidirectional: bool = True,
        num_layers: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.name = (
            f"lstm_{vocab_size}v_{embedding_dim}emb_{hidden_dim}hid"
            f"_{num_class}nc{'_bidirectional' if bidirectional else ''}_{num_layers}lay"
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.padding_idx = padding_idx

        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.drop = torch.nn.Dropout(p=dropout)
        self.fc = torch.nn.Linear(hidden_dim * (2 if bidirectional else 1), num_class)

    def forward(self, padded_text: torch.Tensor, lengths: torch.Tensor|None = None, embedded_input: bool = False, trainable: bool = False) -> torch.Tensor:
        global Embedding_model
        Embedding_model = self.embedding
        if embedded_input:
            text_embedding = padded_text
        else:
            text_embedding = self.embedding(padded_text)

        if lengths == None and trainable == False:
          lengths = torch.tensor([len(padded_text[0])], dtype=torch.int64)

        packed_padded_text = pack_padded_sequence(
            text_embedding, lengths, batch_first=True, enforce_sorted=False
        )

        packed_lstm_output, _ = self.lstm(packed_padded_text)
        lstm_output, _ = pad_packed_sequence(
            packed_lstm_output,
            batch_first=True,
            padding_value=float(self.padding_idx)
            if self.padding_idx is not None
            else 0.0,
        )

        out_forward = lstm_output[:, -1, : self.hidden_dim]
        out_reverse = lstm_output[:, 0, self.hidden_dim :]
        lstm_state = torch.cat((out_forward, out_reverse), 1)

        features: torch.Tensor = self.drop(lstm_state)
        features = self.fc(features)

        if trainable:
          return features
        return features.argmax(1)

def construct_tokenizer(tokenizer_name: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    vocab: torchtext.vocab.Vocab
    if tokenizer_name == "XLMR":
        xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
        xlmr_spm_model_path = (
            r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"
        )

        tokenizer = torchtext.transforms.SentencePieceTokenizer(xlmr_spm_model_path)
        tokenizer.name = "XLMR"
        vocab = load_state_dict_from_url(xlmr_vocab_path)
    else:
        tokenizer = TokenizerModule(tokenizer_name)

        def yield_tokens(
            data_iter: torch.utils.data.IterableDataset[tuple[int, str]]
        ) -> Iterable[list[int]]:
            for _, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(
            yield_tokens(AG_NEWS(split="train")), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
    return tokenizer, vocab

# Call the functions for F training
Embedding_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix seed
torch.manual_seed(0)

tokenizer, vocab = construct_tokenizer('basic_english')
model = 'lstm'
model: torch.nn.Module
if model == "linear":
  model = LinearEmbeddingBagModel(
      vocab_size=len(vocab),
      embedding_dim=64,
      num_class=len(set([label for label, _text in AG_NEWS(split="train")])),
      padding_idx=1,
  )
elif model == "lstm":
  model = SimpleBidirectionalLSTM(
      vocab_size=len(vocab),
      embedding_dim=300,
      hidden_dim=128,
      num_class=len(set([label for label, _text in AG_NEWS(split="train")])),
  )
else:
  raise NotImplementedError

train(
  train_dataset,
  test_dataset,
  model=model,
  tokenizer=tokenizer,
  vocab=vocab,
  batch_size=64,
  epochs=25,
  seed=0,
  learning_rate=1e-5,
)

# Prediction F
import torch
from torchtext.vocab import build_vocab_from_iterator

# Load tokenizer and vocab
tokenizer, vocab = construct_tokenizer('basic_english')
text_transform = torchtext.transforms.Sequential(
    tokenizer,
    torchtext.transforms.VocabTransform(vocab),
    torchtext.transforms.Truncate(256 - 2),
    torchtext.transforms.AddToken(token=0, begin=True),
    torchtext.transforms.AddToken(token=2, begin=False),
)
tokenizer = tokenizer.to(device)
vocab = vocab.to(device)
# Load model
load_path = '/content/models/lstm_95811v_300emb_128hid_4nc_bidirectional_1lay_basic_english_25e_1e-05lr.pt'
model = SimpleBidirectionalLSTM(
    vocab_size=len(vocab),
    embedding_dim=300,
    hidden_dim=128,
    num_class=len(set([label for label, _text in AG_NEWS(split="train")])),
)
model.load_state_dict(torch.load(load_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



# Prediction F
def predict_lstm(text, text_transform, model):
  global device, Trainable

  token_ids = text_transform(text)
  text_lis = [token_ids]
  padded_text = torchtext.functional.to_tensor(text_lis, padding_value=1).to(device)
  # print(padded_text)
  length = [len(token_ids)]
  length_tensor = torch.tensor(length, dtype=torch.int64)

  #print(padded_text, length_tensor)
  prediction = model(padded_text, length_tensor, trainable=False)
  return prediction+1



