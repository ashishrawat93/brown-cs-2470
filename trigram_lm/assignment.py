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
        pass

    def loss(self):
        """
        Build the cross-entropy loss computation graph.
        DO 
        """

        # TODO: Perform the loss computation
        pass

    def optimizer(self):
        """
        Build the training operation, using the cross-entropy loss and an Adam Optimizer.
        """

        # TODO: Execute the training operation
        pass

def read(train_file, dev_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.
    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    """

    # TODO: Read and process text data given paths to train and development data files
    pass 

def main():

    # TODO: Import and process data

    # TODO: Set up placeholders for inputs and outputs to pass into model's constructor

    # TODO: Initialize model and tensorflow variables

    # TODO: Set up the training step, training with 1 epoch and with a batch size of 20

    # TODO: Run the model on the development set and print the final perplexity
    # Remember that perplexity is just defined as: e^(average_loss_per_input)!

if __name__ == "__main__":
    main()
    
