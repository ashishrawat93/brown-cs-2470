"""
Stencil layout for your trigram language model assignment, with embeddings.

The stencil has three main parts:
    - A class stencil for your actual trigram language model. The class is complete with helper
    methods declarations that break down what your model needs to do into reasonably-sized steps,
    which we recommend you use.

    - A "read" helper function to isolate the logic of parsing the raw text files. This is where
    you should build your vocabulary, and transform your input files into data that can be fed into the model.

    - A main-training-block area - this code (under "if __name__==__main__") will be run when the script is,
    so it's where you should bring everything together and execute the actual training of your model.
"""

import tensorflow as tf
import numpy as np
import time

class TrigramLM:
    def __init__(self, X1, X2, Y, vocab_sz):
        """
        Instantiate your TrigramLM Model, with whatever hyperparameters are necessary
        !!!! DO NOT change the parameters of this constructor !!!!

        X1, X2, and Y represent the first, second, and third words of a batch of trigrams.
        (hint: they should be placeholders that can be fed batches via your feed_dict).

        You should implement and use the "read" function to calculate the vocab size of your corpus
        before instantiating the model, as the model should be specific to your corpus.
        """
        
        # TODO: Define network parameters

        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.vocab_sz = vocab_sz
 
        self.logits = self.forward_pass()

        # IMPORTANT - your model MUST contain two instance variables,
        # self.loss and self.train_op, that contain the loss computation graph 
        # (as you will define in in loss()), and training operation (as you will define in train())
        self.loss = self.loss()
        self.train_op = self.optimizer()

    def forward_pass(self):
        """
        Build the inference computation graph for the model, going from the input to the output
        logits (before final softmax activation). This is analogous to "prediction".
        """

        # TODO: Compute the logits
        VOCAB_SZ = self.vocab_sz
        EMBEDDING_SZ = 80

        E = tf.Variable(tf.random_normal([VOCAB_SZ, EMBEDDING_SZ], stddev=0.1))
        W = tf.Variable(tf.random_normal([EMBEDDING_SZ+EMBEDDING_SZ, VOCAB_SZ], stddev=0.1))
        b = tf.Variable(tf.random_normal([VOCAB_SZ], stddev=0.1))

        em1 = tf.nn.embedding_lookup(E, self.X1)
        em2 = tf.nn.embedding_lookup(E, self.X2)
        embedding = tf.concat([em1,em2], axis = 1)

        logits = tf.add(tf.matmul(embedding, W), b)
        # prob = tf.nn.softmax(logits)
        return logits



    def loss(self):
        """
        Build the cross-entropy loss computation graph.
        DO 
        """

        # TODO: Perform the loss computation
        loss = tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.Y)

        # loss = tf.losses.softmax_cross_entropy(self.label, self.Y)
        return loss

    def optimizer(self):
        """
        Build the training operation, using the cross-entropy loss and an Adam Optimizer.
        """

        # TODO: Execute the training operation
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
        return train_op

def read(train_file, dev_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.
    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    """


    train_data = []
    dev_data = []
    with open(train_file, 'r') as f:
        sentences = f.read().split('.\n')

    vocab = set(" ".join(sentences).split())
    word2id = {w: i for i, w in enumerate(list(vocab))}

    s = map(lambda x: x.split(), sentences)

    for each in s:
        for i in range(0, len(each)-2):
            train_data.append([word2id[each[i]], word2id[each[i+1]], word2id[each[i+2]]])
    train_data = np.array(train_data)

    with open(dev_file, 'r') as f:
        sentences = f.read().split('.\n')
    s = map(lambda x: x.split(), sentences)
    for each in s:
        for i in range(0, len(each) - 2):
            dev_data.append([word2id[each[i]], word2id[each[i + 1]], word2id[each[i + 2]]])
    dev_data = np.array(dev_data)

    return (train_data, dev_data, word2id)

def main():

    start_t = time.time()
    data_tuple = read('train.txt', 'dev.txt')
    train_data = data_tuple[0]
    dev_data = data_tuple[1]
    word2id = data_tuple[2]

    size = len(word2id)

    X = tf.placeholder(tf.int32, shape=[None])
    Y = tf.placeholder(tf.int32, shape=[None])
    Z = tf.placeholder(tf.int32, shape=[None])

    model = TrigramLM(X, Y, Z, size)
    loss = model.loss
    train = model.train_op
    curr_loss, step = 0,0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for start, end in zip(range(0, len(train_data) - 20, 20), range(20, len(train_data), 20)):
            l, _ = sess.run( [loss,train],feed_dict={X: train_data[start:end, 0], Y: train_data[start:end, 1], Z: train_data[start:end, 2]})
            # print(curr_loss, step)
            curr_loss, step = curr_loss + l, step + 1

        devlen = len(dev_data)
        loss = sess.run([loss],feed_dict={X:dev_data[0:devlen,0], Y:dev_data[0:devlen,1], Z:dev_data[0:devlen,2]})
        # print(loss)
        loss = loss[0]
        perp = np.exp(loss)
        print("Perplexity is:", perp)

        print("Time :",time.time()-start_t)

if __name__ == "__main__":
    main()
    
