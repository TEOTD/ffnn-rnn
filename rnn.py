import gensim.downloader

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h, vocab_size, pretrained_embeddings=None):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h # int, the size of the hidden dimension
        self.numOfLayer = 2  # Use 2 layers for better representation
        
        # Create embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.embedding_dropout = nn.Dropout(0.5)  # Higher dropout on embeddings
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        # Use LSTM instead of vanilla RNN - much better for long sequences
        self.rnn = nn.LSTM(input_dim, h, self.numOfLayer, dropout=0.5 if self.numOfLayer > 1 else 0, batch_first=False)
        
        # Add an extra hidden layer for better classification
        self.fc1 = nn.Linear(h, h // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Higher dropout
        self.W = nn.Linear(h // 2, 5)
        
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, word_indices):
        # word_indices: (sequence_length, batch_size) - contains word indices
        # [to fill] get embeddings from indices
        embedded = self.embedding(word_indices)  # (sequence_length, batch_size, embedding_dim)
        embedded = self.embedding_dropout(embedded)  # Apply dropout to embeddings
        
        # [to fill] obtain hidden layer representation (LSTM returns output and (hidden, cell))
        rnn_output, (hidden, cell) = self.rnn(embedded)
        
        # Use both last hidden state and mean pooling for richer representation
        # Mean pooling: captures overall sentiment
        mean_pool = torch.mean(rnn_output, dim=0)  # (batch_size, hidden_dim)
        # Last hidden state: captures final context
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Combine both representations
        combined = mean_pool + last_hidden  # Element-wise addition
        
        # [to fill] obtain output layer representations with extra hidden layer
        h1 = self.fc1(combined)
        h1 = self.relu(h1)
        h1 = self.dropout(h1)
        final_output = self.W(h1)
        
        # [to fill] obtain probability dist.
        final_output = self.softmax(final_output)

        return final_output


def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    # Load pretrained GloVe embeddings
    glove_embeds = gensim.downloader.load('glove-twitter-200')
    embedding_dim = glove_embeds.vector_size
    
    # Build vocabulary from training data with preprocessing
    vocab = set()
    for document, _ in train_data:
        for word in document:
            # Apply same preprocessing as during training
            word = word.lower()
            # Remove punctuation to match GloVe format
            word = word.translate(str.maketrans("", "", string.punctuation))
            if word:  # Only add non-empty words
                vocab.add(word)
    
    # Create word to index mapping
    word2idx = {unk: 0}  # Reserve index 0 for unknown words
    idx = 1
    for word in sorted(vocab):
        word2idx[word] = idx
        idx += 1
    
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create embedding matrix initialized with pretrained embeddings
    pretrained_embeddings = torch.zeros(vocab_size, embedding_dim)
    # Initialize with small random values
    pretrained_embeddings.normal_(0, 0.1)
    
    # Fill in pretrained embeddings where available
    found = 0
    for word, idx in word2idx.items():
        if word in glove_embeds:
            pretrained_embeddings[idx] = torch.tensor(glove_embeds[word])
            found += 1
    
    print(f"Loaded pretrained embeddings for {found}/{vocab_size} words")
    
    # Initialize model with pretrained embeddings
    model = RNN(embedding_dim, args.hidden_dim, vocab_size, pretrained_embeddings)
    
    # Freeze embeddings for first 2 epochs to prevent catastrophic forgetting
    model.embedding.weight.requires_grad = False
    
    # Use higher weight decay for stronger L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    best_validation_accuracy = 0
    patience = 3  # Stop if no improvement for 3 epochs (more patience)
    patience_counter = 0

    while not stopping_condition and epoch < args.epochs:
        # Unfreeze embeddings after 2 epochs
        if epoch == 2:
            print("Unfreezing embeddings for fine-tuning...")
            model.embedding.weight.requires_grad = True
            # Recreate optimizer to include embedding parameters
            optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR for fine-tuning
        
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(0, N, minibatch_size)):
            optimizer.zero_grad()
            loss = None
            actual_batch_size = min(minibatch_size, N - minibatch_index)
            for example_index in range(actual_batch_size):
                input_words, gold_label = train_data[minibatch_index + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Convert words to indices (Option 3: using embedding layer)
                word_indices = [word2idx.get(word.lower(), word2idx[unk]) for word in input_words]
                
                # Transform to tensor (sequence_length, batch_size=1)
                word_indices = torch.tensor(word_indices, dtype=torch.long).view(-1, 1)
                output = model(word_indices)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / actual_batch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total


        model.eval()
        correct = 0
        total = 0
        print("Validation started for epoch {}".format(epoch + 1))

        with torch.no_grad():
            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                
                # Convert words to indices (Option 3: using embedding layer)
                word_indices = [word2idx.get(word.lower(), word2idx[unk]) for word in input_words]
                
                # Transform to tensor (sequence_length, batch_size=1)
                word_indices = torch.tensor(word_indices, dtype=torch.long).view(-1, 1)
                output = model(word_indices)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        # Better early stopping: track best validation and use patience
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            patience_counter = 0
            print(f"New best validation accuracy: {best_validation_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            stopping_condition = True
            print("Early stopping triggered!")
            print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
        
        last_validation_accuracy = validation_accuracy
        last_train_accuracy = trainning_accuracy

        epoch += 1



    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
