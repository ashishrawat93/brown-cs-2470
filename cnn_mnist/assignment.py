import os
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:

    def __init__(self, image, label):
        """
        A Model class contains a computational graph that classifies images
        to predictions. Each of its methods builds part of the graph
        on Model initialization. Do not modify the constructor, as doing so
        would break the autograder. You may, however, add class variables
        to use in your graph-building. e.g. learning rate, 

        image: the input image to the computational graph as a tensor
        label: the correct label of an image as a tensor
        prediction: the output prediction of the computational graph,
                    produced by self.forward_pass()
        optimize: the model's optimizing tensor produced by self.optimizer()
        loss: the model's loss produced by computing self.loss_function()
        accuracy: the model's prediction accuracy
        """
        self.image = image
        self.label = label

        # TO-DO: Add any class variables you want to use.

        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        Predicts a label given an image using convolution layers

        :return: the prediction as a tensor
        """
        # TO-DO: Build up the computational graph for the forward pass.
        image = tf.reshape(self.image, [-1, 28, 28, 1])

        kernel1 = tf.Variable(tf.truncated_normal([4,4,1, 32], stddev=0.1))
        bias1 = tf.Variable(tf.truncated_normal([32], stddev=0.1))

        l1 = tf.nn.conv2d(image, filter=kernel1, strides=[1,1,1,1], padding="SAME") + bias1
        l1 = tf.nn.relu(l1)

        pool = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,1,1,1], padding="SAME")


        kernel2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.1))
        bias2 = tf.Variable(tf.truncated_normal([64], stddev=0.1))

        l2 = tf.nn.conv2d(l1, filter=kernel2, strides=[1,2,2,1], padding="SAME") + bias2
        l2 = tf.nn.relu(l2)

        l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # #
        # dimension = l2.get_shape()
        # dim = dimension[1]*dimension[2]*dimension[3]
        # # 3200
        # print(dimension, dim)
        # # print(tf.shape(l2))
        # exit(0)
        # fc_dim = 1568
        fc_dim = 3136

        fc_1 = tf.reshape(l2, shape=[-1, fc_dim])


        W_fc_1_2 =  tf.Variable(tf.truncated_normal([fc_dim, 1000], stddev=0.1))
        b_fc_1_2 = tf.Variable(tf.truncated_normal([1000], stddev=0.1))

        logits_1 = tf.add(tf.matmul(fc_1, W_fc_1_2), b_fc_1_2)

        activation_1 = tf.nn.relu(logits_1)


        W_fc_2_o = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
        b_fc_2_o = tf.Variable(tf.truncated_normal([10], stddev=0.1))

        logits_2 = tf.add(tf.matmul(activation_1, W_fc_2_o), b_fc_2_o)

        prob = tf.nn.softmax(logits_2)

        return prob


    def loss_function(self):
        """
        Calculates the model cross-entropy loss

        :return: the loss of the model as a tensor
        """
        # TO-DO: Add the loss function to the computational graph
        loss = tf.losses.softmax_cross_entropy(self.label, self.prediction)
        return loss

        # loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.prediction), reduction_indices=[1]))
        # return loss

    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer

        :return: the optimizer as a tensor
        """
        # TO-DO: Add the optimizer to the computational graph
        train =  tf.train.AdamOptimizer(learning_rate=0.0002).minimize(self.loss)
        return train

    def accuracy_function(self):
        """
        Calculates the model's prediction accuracy by comparing
        predictions to correct labels – no need to modify this

        :return: the accuracy of the model as a tensor
        """
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():

    # TO-DO: import MNIST data
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # TO-DO: Set-up placeholders for inputs and outputs
    image = tf.placeholder(tf.float32, shape=[None, 784])
    label = tf.placeholder(tf.float32, shape=[None, 10])

    # TO-DO: initialize model and tensorflow variables
    model = Model(image, label)
    # train = model.optim
    start_time = time.time()
    # TO-DO: Set-up the training step, for 2000 batches with a batch size of 50
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for batch in range(2000):
            print("Batch: ", batch)
            train_images, train_labels = data.train.next_batch(batch_size=50)
            sess.run(model.optimize, feed_dict={model.image: train_images, model.label: train_labels})
            print(sess.run(model.accuracy, feed_dict={model.image: train_images, model.label: train_labels}))

        acc = 0
        for _ in range(2000):

            test_images, test_labels = data.test.next_batch(batch_size=50)
            acc += sess.run(model.accuracy, feed_dict={model.image: test_images, model.label: test_labels})

        print("\nTest Accuracy: ", acc / 2000)

        print("Time taken: ", time.time()-start_time)

    # TO-DO: run the model on test data and print the accuracy

    return


if __name__ == '__main__':
    main()
