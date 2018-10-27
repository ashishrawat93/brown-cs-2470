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
import math

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

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    with open(train_file, 'r') as f:
        sentences = f.read().split()

    vocab = set(sentences)
    word2id = {w: i for i, w in enumerate(list(vocab))}

    for each in sentences:
        train_x.append(word2id[each])
        train_y.append(word2id[each])

    trainx = train_x[:-1]
    train_y = train_y[1:]
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    with open(test_file, 'r') as f:
        sentences = f.read().split()

    for each in sentences:
        test_x.append(word2id[each])
        test_y.append((word2id[each]))

    test_x = test_x[:-1]
    test_y = test_y[1:]
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, train_y, test_x, test_y, word2id



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
        EMB_SIZE = 65
        RNN_SIZE = 300

        W = tf.Variable(tf.random_normal([RNN_SIZE, self.vocab_size], stddev=0.1))
        b = tf.Variable(tf.random_normal([self.vocab_size], stddev=0.1))

        E = tf.Variable(tf.random_normal([self.vocab_size, EMB_SIZE],stddev=0.1))
        embs = tf.nn.embedding_lookup(E, self.inputs)

        embs = tf.nn.dropout(embs, self.keep_prob)
        cell = tf.contrib.rnn.GRUCell(RNN_SIZE)
        initstate = cell.zero_state(BATCH_SIZE, tf.float32)
        op, nextState = tf.nn.dynamic_rnn(cell, embs, initial_state = initstate)

        logits = tf.add(tf.tensordot(op, W, axes=[[2],[0]]),b)
        return logits


    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer
        :return: the optimizer as a tensor
        """
        tr = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        return tr


    def loss_function(self):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :return: the loss of the model as a tensor of size 1
        """
        loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(self.prediction, self.labels, tf.ones([BATCH_SIZE,WINDOW_SIZE])))
        return loss

    def perplexity_function(self):
        """
        Calculates the model's perplexity by comparing predictions to correct labels
        :return: the perplexity of the model as a tensor of size 1
        """
        return tf.exp(self.loss)




def main():
    # Preprocess data
    train_file = "train.txt"
    dev_file = "dev.txt"
    train_x, train_y, test_x, test_y, vocab_map = read(train_file, dev_file)

    # TODO: define placeholders
    inputs = tf.placeholder(tf.int32, shape=[None, None])
    outputs = tf.placeholder(tf.int32, shape=[None, None])

    vocab_sz = len(vocab_map)

    # TODO: initialize model
    model = Model(inputs, outputs, 0.7,vocab_sz)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # TODO: Set-up the training step:
    # - 1) divide training set into equally sized batch chunks. We recommend a batch size of 50.
    # - 2) split these batch segments into windows of size WINDOW_SIZE.

    t1_x = []
    t1_y = []
    t2_x = []
    t2_y = []
    # split train data into batches

    datalen = len(train_x)
    numbatches = (datalen // (BATCH_SIZE * WINDOW_SIZE))

    for p1 in range(numbatches):
        tempx = list()
        tempy = list()
        for p2 in range(BATCH_SIZE):
            # print(WINDOW_SIZE*p1)
            l_index1 = (WINDOW_SIZE*p1 + p2*math.floor((datalen-1)/BATCH_SIZE))
            tempx.append(train_x[l_index1 : l_index1 + WINDOW_SIZE])
            tempy.append(train_y[l_index1 : l_index1 + WINDOW_SIZE])
        t1_x.append(tempx)
        t1_y.append(tempy)

    lastrow_input = t1_x[len(t1_x)-1]
    lastrow_input_count = sum([len(row) for row in lastrow_input])

    if not lastrow_input_count == (BATCH_SIZE*WINDOW_SIZE):
        t1_x.pop(len(t1_x)-1)

    lastrow_output = t1_y[len(t1_y) - 1]
    lastrow_output_count = sum([len(row) for row in lastrow_output])

    if not lastrow_output_count == (BATCH_SIZE * WINDOW_SIZE):
        t1_y.pop(len(t1_y) - 1)

    # print(len(train_x))
    # print(len(train_x)/(BATCH_SIZE*WINDOW_SIZE))

    datalen = len(test_x)
    numbatches = (datalen // (BATCH_SIZE * WINDOW_SIZE))

    for p1 in range(numbatches):
        tempx = list()
        tempy = list()
        for p2 in range(BATCH_SIZE):
            l_index1 = (WINDOW_SIZE * p1 + p2 * math.floor((datalen - 1) / BATCH_SIZE))
            tempx.append(test_x[l_index1: l_index1 + WINDOW_SIZE])
            tempy.append(test_y[l_index1: l_index1 + WINDOW_SIZE])
        t2_x.append(tempx)
        t2_y.append(tempy)

    # print(t2_x, t2_y)

    lastrow_input = t2_x[len(t2_x) - 1]
    lastrow_input_count = sum([len(row) for row in lastrow_input])

    if not lastrow_input_count == (BATCH_SIZE * WINDOW_SIZE):
        t2_x.pop(len(t2_x) - 1)

    lastrow_output = t2_y[len(t2_y) - 1]
    lastrow_output_count = sum([len(row) for row in lastrow_output])

    if not lastrow_output_count == (BATCH_SIZE * WINDOW_SIZE):
        t2_y.pop(len(t2_y) - 1)

    for idx in range(len(t1_x)):
        train = sess.run([model.optimize], feed_dict = {inputs: t1_x[idx], outputs: t1_y[idx]})
        # print(loss)

    # TODO: Run the model on the development set and print the final perplexity

    loss_sum = []
    for idx in range(len(t2_x)):
        loss_sum.append(sess.run([model.perplexity], feed_dict = {inputs: t2_x[idx], outputs: t2_y[idx]})[0])

    print("Perplexity is ", (sum(loss_sum)/len(t2_x)))

if __name__ == '__main__':
    main()
