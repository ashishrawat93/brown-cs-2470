import os
import tensorflow as tf
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

    # TODO: import MNIST data
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)


    # TODO: Set-up placeholders for inputs and outputs
    batch_size = 200
    input_image = tf.placeholder(tf.float32, [batch_size, 784])
    labels = tf.placeholder(tf.float32, [batch_size, 10])



    # TODO: initialize model and tensorflow variables

    weights = tf.Variable(tf.random_normal([784, 10], stddev=0.1))
    bias = tf.Variable(tf.random_normal([10], stddev=0.1))

    probs = tf.nn.softmax(tf.matmul(input_image, weights) + bias)
    loss = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(probs), reduction_indices=[1]))


    # TODO: Set-up the training step, for as many of the 60,000 examples as you'd like
    #     where the batch size is greater than 1
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    correct_predictions = tf.equal(tf.argmax(probs,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # TODO: run the model on test data and print the accuracy
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(40000):
        images, correct_labels = data.train.next_batch(batch_size)
        sess.run(train, feed_dict={input_image:images, labels:correct_labels})


    tot_acc = 0
    for i in range(1000):
        images, correct_labels = data.train.next_batch(batch_size)
        tot_acc = sess.run(accuracy, feed_dict={input_image:images, labels:correct_labels})

    print("Test Accuracy: ", tot_acc/1000)
    return


if __name__ == '__main__':
    main()
