
!pip install --upgrade torchtext

!pip install portalocker

!pip install captum

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
train_dataset = [Train_dataset[i] for i in range(3000)] # Choose a subsample of Train_dataset
Test_dataset = to_map_style_dataset(test_datapipe)
test_dataset = [Test_dataset[i] for i in range(300)] # Choose a subsample of Test_dataset

# Create binary vectors
def binary_vectors(corpus):
  vectorizer = CountVectorizer()
  document_vector = vectorizer.fit_transform(corpus)
  words = vectorizer.get_feature_names_out()
  return document_vector.toarray(), words

# Create characteristic vetors for the corpus
def characteristic_vector_creator_original(corpus):
  vectorizer = CountVectorizer()
  document_vector = vectorizer.fit_transform(corpus)
  words = vectorizer.get_feature_names_out()
  return document_vector.toarray(), words

# Create characteristic vetors for z_1 (simple representation of the text)
def characteristic_vector_creator_purturbed(z,corpus):
  vectorizer = CountVectorizer()
  vectorizer.fit(corpus)
  document_vector = vectorizer.transform([z])
  return document_vector.toarray()

# Sampling in the token space 
def sampling(corpus, doc_index):
  list_corpus = corpus[doc_index].split(' ')
  num_indices = int(random.uniform(0, len(list_corpus)))
  random_indices = random.sample(range(len(list_corpus)), num_indices)
  random_indices.sort()  # Sort the indices in ascending order
  list_corpus = [list_corpus[i] for i in random_indices]
  corpus = ' '.join(list_corpus)
  return corpus

# Class of simple Linear model G
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
def linear_model_g(X_numpy,y_numpy,weights,learning_rate,num_epoch):
  X = torch.from_numpy(X_numpy.astype(np.float32))
  y = torch.from_numpy(y_numpy.astype(np.float32))
  y = y.view(y.shape[0],1)
  weights = torch.from_numpy(np.array(weights).astype(np.float32))

  n_samples,n_features = X.shape

  # 1)Model
  input_size = n_features
  output_size = 1
  model = nn.Linear(input_size,output_size)
  """
  model = LinearRegression(input_size, output_size)
  """
  # 2)Loss and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  # 3)Training loop
  for epoch in range(num_epoch):
    # forward path
    y_pred = model(X)
    # Loss
    loss = criterion(y_pred,y)
    weighted_mse_loss = torch.mean(loss * weights)
    # Backward path
    weighted_mse_loss.backward()
    # Update
    optimizer.step()
    # Empty grad
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epoch}, Loss: {weighted_mse_loss.item()}")
  # Class of simple Linear model G
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)
def linear_model_g(X_numpy,y_numpy,weights,learning_rate,num_epoch):
  X = torch.from_numpy(X_numpy.astype(np.float32))
  y = torch.from_numpy(y_numpy.astype(np.float32))
  y = y.view(y.shape[0],1)
  weights = torch.from_numpy(np.array(weights).astype(np.float32))

  n_samples,n_features = X.shape

  # 1)Model
  input_size = n_features
  output_size = 1
  model = nn.Linear(input_size,output_size)
  """
  model = LinearRegression(input_size, output_size)
  """
  # 2)Loss and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  # 3)Training loop
  for epoch in range(num_epoch):
    # forward path
    y_pred = model(X)
    # Loss
    loss = criterion(y_pred,y)
    weighted_mse_loss = torch.mean(loss * weights)
    # Backward path
    weighted_mse_loss.backward()
    # Update
    optimizer.step()
    # Empty grad
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epoch}, Loss: {weighted_mse_loss.item()}")

  # Get the final weights
  final_weights = model.weight.detach().numpy()
  final_bias = model.bias.detach().numpy()
  predicted = model(X).detach().numpy()
  return final_weights


# Model F here we used bidirectinal LSTM
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
  batch_size=20,
  epochs=5,
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
load_path = '/content/models/lstm_95811v_300emb_128hid_4nc_bidirectional_1lay_basic_english_5e_1e-05lr.pt'
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
  # length = [len(token_ids)]
  # length_tensor = torch.tensor(length, dtype=torch.int64)

  #print(padded_text, length_tensor)
  prediction = model(padded_text)
  return prediction+1

text = train_dataset[3][1]
predict_lstm(text, text_transform, model)

# Distance measure PI in the token space (binary vectors)
def dist_pi_binary(x_1,z_1):
    width = 0.05
    dist = cosine_distances(x_1, z_1)
    weight = np.exp(-(dist**2)/width**2)
    return weight


# Binary vector creator
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class BinaryVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vocabulary_ = {}
        self.sentences_ = []

    def fit(self, X, y=None):
        self.vocabulary_ = {}
        unique_words = set()
        self.sentences_ = X  # Store the original sentences
        for doc in X:
            unique_words.update(doc.split())
        self.vocabulary_ = {word: i for i, word in enumerate(unique_words)}
        return self

    def transform(self, X):
        binary_vectors = np.zeros((len(X), len(self.vocabulary_)))
        word_vectors = []
        for i, doc in enumerate(X):
            words = doc.split()
            word_vectors.append(words)
            for j, word in enumerate(words):
                if word in self.vocabulary_:
                    binary_vectors[i, self.vocabulary_[word]] = 1
        return binary_vectors, word_vectors

def match_binary_vector(binary_vector, elements_list):
    matched_elements = [elements_list[i] for i in np.where(binary_vector == 1)[0]]
    matched_string = ' '.join(matched_elements)
    return matched_string

# build the setup for token space 
# testing 
index = 3	# Which doc to be explained 
x = train_dataset[index]
corpus = [item[1] for item in train_dataset]
text = [corpus[index]]
lis_w,lis_label_f,purturbed_sample = [],[],[]
num_puturb = 8
# For each point to be explained create 10 perturbed binary vectors
for i in range(num_puturb):
  z = sampling(corpus,index)
  text.append(z)
# Fit the vectorizer on the text
vectorizer = BinaryVectorizer()
binary_vectors , word_vectors = vectorizer.fit_transform(text)
# Recreate the actual sentence, and return label f
for i in range(1,len(word_vectors)):
  z = match_binary_vector(binary_vectors[i],word_vectors[0])
  purturbed_sample.append(z)
  label_f = predict_lstm(z, text_transform, model)
  lis_label_f.append(label_f)
lis_z_1 = binary_vectors.copy()
# Weight
for i in range(1,num_puturb):
  lis_w.append(dist_pi_binary([lis_z_1[0]],[lis_z_1[i]])[0][0])
lis_f = []
for i in range(len(lis_label_f)):
  lis_f.append(lis_label_f[i].item())
# Train g
final_weights = linear_model_g(lis_z_1[1:],np.array(lis_f),lis_w,learning_rate=0.01,num_epoch=100)

def find_mostimportant_words(lst):
    abs_lst = np.abs(lst)  # Take the absolute values of the list
    indices = np.argsort(abs_lst[0])  # Get the indices of the three largest absolute values last 5 : np.argsort(abs_lst[0])[:-5]
    indices = np.flip(indices)
    # Return the three largest absolute values and their indices
    return lst[0][indices], indices


largest_values, indices = find_mostimportant_words(final_weights)

# Print the explanation
explanation_implemented_lime = dict(zip(np.array(word_vectors[0])[indices],largest_values))

largest_values, indices = find_mostimportant_words(final_weights)
indices

# Test against Captum

from captum.attr import Lime
device = 'cpu'
model = model.to(device)
token_ids = text_transform(text[0])
text_lis = [token_ids]
length = [len(token_ids)]
length_tensor = torch.tensor(length, dtype=torch.int64)
padded_text = torchtext.functional.to_tensor(text_lis, padding_value=1).to(device)
captum_lime = Lime(model)
captum_lime_attr = captum_lime.attribute(padded_text)
lime_words = []
for i in token_ids:
  lime_words.append(vocab.get_itos()[i])
explanation_captum_lime = dict(zip(lime_words,captum_lime_attr[0].numpy()))

def validate_against_captum(explanation_implemented_lime,explanation_captum_lime,elements):
  """
  gets both explanations and check if they are close enough, also checks in the first n elements if there exists same words
  """

  explanation_implemented_lime = {key.lower(): value for key, value in explanation_implemented_lime.items()}

  # First test
  common_keys = set(explanation_implemented_lime.keys()) & set(explanation_captum_lime.keys())
  # Convert dictionary values to tensors for the common keys
  tensor_imp_lime = torch.tensor([explanation_implemented_lime[key] for key in common_keys])
  tensor_cap_lime = torch.tensor([explanation_captum_lime[key] for key in common_keys])
  # Perform element-wise closeness comparison
  are_close = torch.isclose(tensor_imp_lime, tensor_cap_lime, atol=1).all()

  # Second test
  # sort based on weights
  sorted_values_dict1 = sorted(explanation_implemented_lime.values(),reverse=True)
  sorted_values_dict2 = sorted(explanation_captum_lime.values(),reverse=True)
  # Check how many of the keys are the same
  if len(explanation_captum_lime) < elements or len(explanation_implemented_lime) < elements :
    elements = min(len(explanation_implemented_lime), len(explanation_captum_lime))
  else:
    pass
  common_elements = set(list(explanation_implemented_lime.keys())[:elements]).intersection(set(list(explanation_captum_lime.keys())[:elements]))
  common_elements_counts = len(common_elements)

  return are_close,common_elements_counts

# test the fkt
are_close,common_elements_counts = validate_against_captum(explanation_implemented_lime,explanation_captum_lime,elements=18)
print(are_close,common_elements_counts)

# remove most important words from the data
# go over all the test documents and find their 5 most important words
# randomly chose documents
import random
import heapq

def random_test_docs(num_random):
  indices = []
  for _ in range(num_random):
      index = random.randint(0, 120000)
      indices.append(index)
  return indices

def find_5_mostimportant_words(lst):
    abs_lst = np.abs(lst)  # Take the absolute values of the list
    indices = np.argsort(abs_lst[0])[:-5]  # Get the indices of the three largest absolute values last 5 : np.argsort(abs_lst[0])[:-5]
    indices = np.flip(indices)
    # Return the three largest absolute values and their indices
    return lst[0][indices], indices


word_counts = {}
indices = random_test_docs(3)
indices =[0,1,2,3,4,5]
for ind in indices:
  index = ind
  x = train_dataset[index]
  corpus = [item[1] for item in train_dataset]
  text = [corpus[index]]
  lis_w,lis_label_f,purturbed_sample = [],[],[]
  num_puturb = 8
  # For each point to be explained create 10 perturbed binary vectors
  for i in range(num_puturb):
    z = sampling(corpus,index)
    text.append(z)
  # Fit the vectorizer on the text
  vectorizer = BinaryVectorizer()
  binary_vectors , word_vectors = vectorizer.fit_transform(text)
  # Recreate the actual sentence, and return label f
  for i in range(1,len(word_vectors)):
    z = match_binary_vector(binary_vectors[i],word_vectors[0])
    purturbed_sample.append(z)
    label_f = predict_lstm(z, text_transform, model)
    lis_label_f.append(label_f)
  lis_z_1 = binary_vectors.copy()
  # Weight
  for i in range(1,num_puturb):
    lis_w.append(dist_pi_binary([lis_z_1[0]],[lis_z_1[i]])[0][0])
  lis_f = []
  for i in range(len(lis_label_f)):
    lis_f.append(lis_label_f[i].item())
  # Train g
  final_weights = linear_model_g(lis_z_1[1:],np.array(lis_f),lis_w,learning_rate=0.01,num_epoch=100)
  largest_values, indices = find_5_mostimportant_words(final_weights)
  words = np.array(word_vectors[0])[indices]
  for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1




# count most important words
most_important_words = heapq.nlargest(5, word_counts.items(), key=lambda x: x[1])
most_important_words = dict(most_important_words)
print(most_important_words)

# Replace important word with token
import torchtext
index = 2
word_to_replace = list(most_important_words.keys())[index]
padding_token = "[PAD]"

# Create a new train and test dataset with pad tokens
modified_train_dataset = [(i[0], i[1].replace(word_to_replace, padding_token)) for i in Train_dataset]
modified_test_dataset = [(i[0], i[1].replace(word_to_replace, padding_token)) for i in Test_dataset]


# Retrain f and check the models accuracy

Embedding_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# fix seed
torch.manual_seed(0)

tokenizer, vocab = construct_tokenizer('basic_english')
model = 'lstm'
model: torch.nn.Module
model = SimpleBidirectionalLSTM(
    vocab_size=len(vocab),
    embedding_dim=300,
    hidden_dim=128,
    num_class=len(set([label for label, _text in AG_NEWS(split="train")])),
)

train(
  modified_train_dataset,
  modified_test_dataset,
  model=model,
  tokenizer=tokenizer,
  vocab=vocab,
  batch_size=20,
  epochs=5,
  seed=0,
  learning_rate=1e-5,
)

# LIME with Model Centric Embedding Space implementation
import numpy as np

# Create word embeddings based on the model
def word_embedding(x,text_transform, model):
    global device
    token_ids_x = text_transform(x)
    lis_x = [token_ids_x]
    padded_text_x = torchtext.functional.to_tensor(lis_x, padding_value=1)
    embedding_x = model.embedding(padded_text_x).cpu()
    embedding_x = torch.mean(embedding_x, dim=1).detach().numpy()
    return embedding_x

# Define baseline vector
embedding_dim = 300
baseline_vector = np.zeros(embedding_dim)

# Add Gaussian noise to the input text using baseline
def add_gaussian_noise(input_text, baseline_vector, std_dev=0.1):
    noise_vector = np.random.normal(loc=0.0, scale=std_dev, size=embedding_dim)
    embedding = word_embedding(word, text_transform, model)
    perturbed_word_embedding =  embedding + baseline_vector + noise_vector


    return perturbed_word_embedding

def dist_embedding(x,z,text_transform, model):
    word_embedding_x = word_embedding(x,text_transform, model)
    word_embedding_z = word_embedding(z,text_transform, model)
    width = 0.05
    dist = cosine_distances(word_embedding_x, word_embedding_z)
    weight = np.exp(-(dist**2)/width**2)
    return weight

# testing with embedding space
num_puturb = 10
index = 3
corpus = [item[1] for item in train_dataset]
input_text = [corpus[index]]
perturbed_text = add_gaussian_noise(input_text, baseline_vector, std_dev=0.1)
x = input_text
lis_z, lis_z_1, lis_w,lis_label_f = [],[],[],[]
for i in range(num_puturb):
  # For each point to be explained create 10 perturbed samples
    # z
    z = add_gaussian_noise(input_text, baseline_vector, std_dev=0.1)
    # simple rep z ????????????????????
    z_1 = characteristic_vector_creator_purturbed(z,corpus)[0]
    # Compute weights in embedding space
    weight = dist_embedding(x,z,text_transform, model)[0]
    # Predict label f ????????????????????
    label_f = predict_lstm(z, text_transform, model)

    lis_z.append(z)
    lis_z_1.append(z_1)
    lis_w.append(weight)
    lis_label_f.append(label_f)
lis_f = []
for i in range(len(lis_label_f)):
  lis_f.append(lis_label_f[i].item())
# Train g
final_weights = linear_model_g(np.array(lis_z_1),np.array(lis_f),lis_w,learning_rate=0.01,num_epoch=1000)


