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
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.dropout = nn.Dropout(0.5)  # Add dropout to prevent overfitting
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax(dim=-1) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # Obtain first hidden layer representation
        hidden_layer = self.activation(self.W1(input_vector))
        # Apply dropout to hidden layer to prevent overfitting
        hidden_layer = self.dropout(hidden_layer)
        # Obtain output layer representation
        output_layer = self.W2(hidden_layer)
        # Obtain probability distribution
        predicted_vector = self.softmax(output_layer)
        return predicted_vector


# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word.lower())  # Lowercasing reduces vocab size and improves generalization
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word.lower(), word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data


# Computes the Inverse Document Frequency (IDF) for each word
def compute_idf(training_data, word2index):
    N = len(training_data)
    idf = torch.zeros(len(word2index))
    
    for document, _ in training_data:
        unique_words = set(word.lower() for word in document)
        for word in unique_words:
            if word in word2index:
                idf[word2index[word]] += 1
    
    # IDF = log(N / (df + 1)) where df is document frequency
    # Adding 1 to avoid division by zero
    idf = torch.log(N / (idf + 1))
    return idf


# Converts data to TF-IDF representation
def convert_to_tfidf_representation(data, word2index, idf):
    tfidf_data = []
    for document, y in data:
        # Compute term frequency (TF)
        tf = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word.lower(), word2index[unk])
            tf[index] += 1
        
        # Apply TF-IDF weighting
        tfidf = tf * idf
        
        # L2 normalization for better performance
        norm = torch.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm
        
        tfidf_data.append((tfidf, y))
    return tfidf_data


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

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    # Compute IDF from training data
    idf = compute_idf(train_data, word2index)
    # Convert to TF-IDF representation
    train_data = convert_to_tfidf_representation(train_data, word2index, idf)
    valid_data = convert_to_tfidf_representation(valid_data, word2index, idf)

    # train_data = convert_to_vector_representation(train_data, word2index)
    # valid_data = convert_to_vector_representation(valid_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    # Use Adam with weight decay for better regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    
    # Early stopping with stricter patience
    best_val_acc = 0
    patience = 1
    patience_counter = 0
    
    # Track training history for learning curves
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        epoch_loss = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(0, N, minibatch_size)):
            optimizer.zero_grad()
            loss = None

            actual_batch = train_data[minibatch_index: minibatch_index + minibatch_size]
            actual_batch_size = len(actual_batch)

            for example_index in range(actual_batch_size):
                input_vector, gold_label = actual_batch[example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / actual_batch_size
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = epoch_loss / (N // minibatch_size)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, train_acc))
        print("Training loss for epoch {}: {:.4f}".format(epoch + 1, avg_train_loss))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        model.eval()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16
        N = len(valid_data)
        with torch.no_grad():
            for minibatch_index in tqdm(range(0, N, minibatch_size)):
                loss = None

                actual_batch = valid_data[minibatch_index: minibatch_index + minibatch_size]
                actual_batch_size = len(actual_batch)

                for example_index in range(actual_batch_size):
                    input_vector, gold_label = actual_batch[example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / actual_batch_size
        print("Validation completed for epoch {}".format(epoch + 1))
        val_acc = correct / total
        val_accuracies.append(val_acc)
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_acc))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Best validation accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping Best validation: {best_val_acc:.4f}")
                break
    
    # Save learning curves to file for plotting
    history = {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'best_val_accuracy': best_val_acc
    }
    with open(f'ffnn_history_h{args.hidden_dim}.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Learning curves saved to ffnn_history_h{args.hidden_dim}.json")