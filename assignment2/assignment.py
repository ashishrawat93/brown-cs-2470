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
        self.image =  image
        self.label = label

        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()



    def forward_pass(self):
        """
        Predicts a label given an image using fully connected layers

        :return: the predicted label as a tensor
        """



        weights1 = tf.Variable(tf.random_normal([784, 100], stddev=0.1))
        bias1 = tf.Variable(tf.random_normal([100], stddev=0.1))

        weights2 = tf.Variable(tf.random_normal([100, 10], stddev=0.1))
        bias2 = tf.Variable(tf.random_normal([10], stddev=0.1))

        logits_1 = tf.add(tf.matmul(self.image, weights1), bias1)
        activation_1 = tf.nn.relu(logits_1)

        logits_2 = tf.add(tf.matmul(activation_1, weights2), bias2)
        activation_2 = tf.nn.softmax(logits_2)
        return activation_2

    def loss_function(self):
        """
        Calculates the model loss

        :return: the loss of the model as a tensor
        """
        # TODO replace pass with loss_function method
        loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.prediction), reduction_indices=[1]))
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.label))
        return loss
    def optimizer(self):
        """
        Optimizes the model loss

        :return: the optimizer as a tensor
        """
        # TODO replace pass with optimizer method
        train = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(self.loss)
        return train

    def accuracy_function(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():

    data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    image =  tf.placeholder(tf.float32, shape=[None, 784])
    label = tf.placeholder(tf.float32, shape=[None, 10])

    model = Model(image, label)

    train = model.optimize

    # correct_prediction = tf.equal(tf.argmax(model.prediction, 1),
    #                               tf.argmax(model.label, 1))
    #
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for b in range(2000):
    #         inp, lab = data.train.next_batch(batch_size=100)
    #
    #         sess.run(train,feed_dict={model.image:inp, model.label:lab})
    #         acc = sess.run(accuracy,feed_dict={model.image:inp, model.label:lab})
    #         print("\nAccuracy for batch \'",b+1, "\' is :",acc )
    #
    #     acc = 0
    #     for _ in range(2000):
    #         timg, tlab = data.test.next_batch(batch_size=100)
    #         acc += sess.run(accuracy, feed_dict={model.image:timg, model.label:tlab})
    #
    #     print("\nTest Accuracy: ", acc/2000)

if __name__ == '__main__':
    main()
