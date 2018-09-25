import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.
class Model:

    def __init__(self, image, label):
        """
        A Model class contains a computational graph that classifies images
        to predictions. Each of its methods builds part of the graph
        on Model initialization.

        image: the input image to the computational graph as a tensor
        label: the correct label of an image as a tensor
        prediction: the output prediction of the computational graph,
                    produced by self.forward_pass()
        optimize: the model's optimizing tensor produced by self.optimizer()
        loss: the model's loss produced by computing self.loss_function()
        accuracy: the model's prediction accuracy – no need to modify this
        """
        self.image = image
        self.label = label
        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()
        # self.learn_rate = 0.01
        # self.batch_size = 100


    def forward_pass(self):
        """
        Predicts a label given an image using fully connected layers

        :return: the predicted label as a tensor
        """
        # TODO replace pass with forward_pass method
        pass

    def loss_function(self):
        """
        Calculates the model loss

        :return: the loss of the model as a tensor
        """
        # TODO replace pass with loss_function method
        pass

    def optimizer(self):
        """
        Optimizes the model loss

        :return: the optimizer as a tensor
        """
        # TODO replace pass with optimizer method
        pass

    def accuracy_function(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():


    data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    tf.set_random_seed(5)
    batch_size = 100
    input_image = tf.placeholder(tf.float32, [batch_size, 784])
    y = tf.placeholder(tf.int64, [batch_size, 10])





    weights = tf.Variable(tf.random_normal([784, 400]))
    bias = tf.Variable(tf.random_normal([400]))
    # bias = tf.Variable(tf.zeros([500]))

    weights2 = tf.Variable(tf.random_normal([400,10]))
    bias2 = tf.Variable(tf.random_normal([10]))
    # bias2 = tf.Variable(tf.zeros([10]))



    logits_1 = tf.add( tf.matmul(input_image, weights), bias)
    activation_1 = tf.nn.sigmoid(logits_1)

    logits_ = tf.add(tf.matmul(activation_1, weights2), bias2)
    predction = tf.argmax(tf.nn.sigmoid(logits_), axis=1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_, labels=y))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        # sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for i in range(2000):

            images, correct_labels = data.train.next_batch(batch_size)
            images =  images/255
            # print(correct_labels.shape, correct_labels[0])
            # exit(0)
            sess.run(train, feed_dict={input_image:images, y:correct_labels})

        tot_acc = 0
        for i in range(2000):
            images, correct_labels = data.test.next_batch(batch_size)
            images = images/255
            # print(images.shape, labels.shape)
            # break
            # intt = sess.run(accuracy, feed_dict={input_image:images, y:correct_labels})
            # tot_acc +=intt
            #
            # print("\nTest Accuracy: ", intt, "\n___________________\n")

            print(sess.run(predction))
            sess.run(logits_)

        print("\nFINAL Test Accuracy: ", tot_acc/2000, "\n_________________\n")


    return


if __name__ == '__main__':
    main()
