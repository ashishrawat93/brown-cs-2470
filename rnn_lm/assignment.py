"""
Stencil layout for your RNN language model assignment.
The stencil has three main parts:
    - A class stencil for your language model
    - A "read" helper function to isolate the logic of parsing the raw text files. This is where
    you should build your vocabulary, and transform your input files into data that can be fed into the model.
    - A main-training-block area - this code (under "if __name__==__main__") will be run when the script is,
    so it's where you should bring everything together and execute the actual training of your model.


Q: What did the computer call its father?
A: Data!

"""

import tensorflow as tf
import numpy as np


# Use this variable to declare your batch size. Do not rename this variable.
BATCH_SIZE = 50

# Your window size must be 20. Do not change this variable!
WINDOW_SIZE = 20


def read(train_file, test_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.

    !!!!!PLEASE FOLLOW THE STENCIL. WE WILL GRADE THIS!!!!!!!

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of (train_x, train_y, test_x, test_y, vocab)

    train_x: List of word ids to use as training input
    train_y: List of word ids to use as training labels
    test_x: List of word ids to use as testing input
    test_y: List of word ids to use as testing labels
    vocab: A dict mapping from word to vocab id
    """

    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.

    pass


class Model:
    def __init__(self, inputs, labels, keep_prob, vocab_size):
        """
        The Model class contains the computation graph used to predict the next word in sequences of words.

        Do not delete any of these variables!

        inputs: A placeholder of input words
        label: A placeholder of next words
        keep_prob: The keep probability of dropout in the embeddings
        vocab_size: The number of unique words in the data
        """

        # Input tensors, DO NOT CHANGE
        self.inputs = inputs
        self.labels = labels
        self.keep_prob = keep_prob

        # DO NOT CHANGE
        self.vocab_size = vocab_size
        self.prediction = self.forward_pass()  # Logits for word predictions
        self.loss = self.loss_function()  # The average loss of the batch
        self.optimize = self.optimizer()  # An optimizer (e.g. ADAM)
        self.perplexity = self.perplexity_function()  # The perplexity of the model, Tensor of size 1

    def forward_pass(self):
        """
        Use self.inputs to predict self.labels.
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :return: logits: The prediction logits as a tensor
        """
        pass

    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer
        :return: the optimizer as a tensor
        """
        pass

    def loss_function(self):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :return: the loss of the model as a tensor of size 1
        """
        pass

    def perplexity_function(self):
        """
        Calculates the model's perplexity by comparing predictions to correct labels
        :return: the perplexity of the model as a tensor of size 1
        """
        pass


def main():
    # Preprocess data
    train_file = ""
    dev_file = ""
    train_x, train_y, test_x, test_y, vocab_map = read(train_file, dev_file)

    # TODO: define placeholders

    # TODO: initialize model

    # TODO: Set-up the training step:
    # - 1) divide training set into equally sized batch chunks. We recommend a batch size of 50.
    # - 2) split these batch segments into windows of size WINDOW_SIZE.

    # TODO: Run the model on the development set and print the final perplexity


if __name__ == '__main__':
    main()
