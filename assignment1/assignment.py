import numpy as np
import os
import gzip
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:
    """
        This model class provides the data structures for your NN, 
        and has functions to test the three main aspects of the training.
        The data structures should not change after each step of training.
        You can add to the class, but do not change the 
        stencil provided except for the blanks and pass statements.
        Make sure that these functions work with a loop to call them multiple times,
        instead of implementing training over multiple steps in the function

        Arguments: 
        train_images - NumPy array of training images
        train_labels - NumPy array of labels
    """
    def __init__(self, train_images, train_labels):
        input_size, num_classes, batchSz, learning_rate = 784, 10, 1, 0.5
        self.train_images = train_images/255
        self.train_labels = train_labels
        self.learn_rate = 0.5
        self.num_classes = num_classes
        self.input_size = input_size

        # sets up weights and biases...
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros(num_classes)



    def run(self):
        """
        Does the forward pass, loss calculation, and back propagation
        for this model for one step

        Args: None
        Return: None
        """
        # get a random sample
        idx = np.random.randint(low=0, high=self.train_images.shape[0])
        sample = self.train_images[idx]
        label = self.train_labels[idx]

        # forward pass
        activation = np.dot(sample, self.W) + self.b
        predicted = self.softmax(activation)

        # backprop
        error = predicted - self.one_hot(label)
        x = sample.reshape(-1, 1)
        gradient = x * error

        # weight update
        self.W = self.W - self.learn_rate * gradient
        self.b = self.b - self.learn_rate * error


    def softmax(self, predicted):
        """
        Calculates the softmax activation across 10 classes
        :param predicted:
        :return:
        """
        return np.exp(predicted)/ sum(np.exp(predicted))


    def cross_entropy(self, label, predicted):
        """
        Calculates the cross entorpy loss
        :param label: an int depecting the digit
        :param predicted: the digit predicted by the model
        :return: returns the cross_entropy loss
        """
        return -1 * np.log(predicted[label])


    def one_hot(self, label):
        """
        Calculates one hot representation for each label
        :param label: an int depicting the digit
        :return: returns an 1D nparray -  the one hot representation for a given class label
        """
        x = np.zeros(self.num_classes)
        x[label] = 1
        return x


    def accuracy_function(self, test_images, test_labels):
        """
        Calculates the accuracy of the model against test images and labels

        DO NOT EDIT
        Arguments
        test_images: a normalized NumPy array
        test_labels: a NumPy array of ints
        """
        scores = np.dot(test_images, self.W) + self.b
        predicted_classes = np.argmax(scores, axis=1)
        return np.mean(predicted_classes == test_labels)


def load_data():

    train_images = []
    with open('train-images-idx3-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        for _ in range(60000):
            stream = bytestream.read(784)
            train = np.frombuffer(stream, dtype=np.uint8)

            train_images.append(train)
    train_images = np.array(train_images)


    train_labels = []
    with open('train-labels-idx1-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        for _ in range(60000):
            stream = bytestream.read(1)
            labels = np.frombuffer(stream, dtype=np.uint8)

            train_labels.append(labels)
    train_labels = np.array(train_labels).reshape(-1)

    test_images = []
    with open('t10k-images-idx3-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(16)
        for _ in range(10000):
            stream = bytestream.read(784)
            train = np.frombuffer(stream, dtype=np.uint8)

            test_images.append(train)
    test_images = np.array(test_images)

    test_labels = []
    with open('t10k-labels-idx1-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        bytestream.read(8)
        for _ in range(10000):
            stream = bytestream.read(1)
            train = np.frombuffer(stream, dtype=np.uint8)

            test_labels.append(train)
    test_labels = np.array(test_labels).reshape(-1)


    return train_images, train_labels, test_images, test_labels

def main():
    # TO-DO: import MNIST test data

    train_images, train_labels, test_images, test_labels = load_data()

    # normalize data
    test_images = test_images / 255

    model = Model(train_images, train_labels)

    for _ in range(10000):
        model.run()

    print(model.accuracy_function(test_images, test_labels))


if __name__ == '__main__':
    main()
