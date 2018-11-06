import numpy as np
import tensorflow as tf
import math

TRAIN_FR = "french_train.txt"
TRAIN_EN = "english_train.txt"
TEST_FR = "french_test.txt"
TEST_EN = "english_test.txt"

# This variable is the batch size the auto-grader will use when training your model.
BATCH_SIZE = 100

# Do not change these variables.
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 12
STOP_TOKEN = "*STOP*"


def pad_corpus(french_file_name, english_file_name):
    """
    Arguments are files of French, English sentences. All sentences are padded with "*STOP*" at
    the end to make their lengths match the window size. For English, an additional "*STOP*" is
    added to the beginning. For example, "I am hungry ." becomes
    ["*STOP*, "I", "am", "hungry", ".", "*STOP*", "*STOP*", "*STOP*",  "*STOP", "*STOP", "*STOP", "*STOP", "*STOP"]

    :param french_file_name: string, a path to a french file
    :param english_file_name: string, a path to an english file

    :return: A tuple of: (list of padded sentences for French, list of padded sentences for English, list of French sentence lengths, list of English sentence lengths)
    """

    french_padded_sentences = []
    french_sentence_lengths = []
    with open(french_file_name, 'rt', encoding='latin') as french_file:
        for line in french_file:
            padded_french = line.split()[:FRENCH_WINDOW_SIZE]
            french_sentence_lengths.append(len(padded_french))
            padded_french += [STOP_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_french))
            french_padded_sentences.append(padded_french)

    english_padded_sentences = []
    english_sentence_lengths = []
    with open(english_file_name, "rt", encoding="latin") as english_file:
        for line in english_file:
            padded_english = line.split()[:ENGLISH_WINDOW_SIZE]
            english_sentence_lengths.append(len(padded_english))
            padded_english = [STOP_TOKEN] + padded_english + [STOP_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_english))
            english_padded_sentences.append(padded_english)

    return french_padded_sentences, english_padded_sentences, french_sentence_lengths, english_sentence_lengths


class Model:
    """
        This is a seq2seq model.

        REMINDER:

        This model class provides the data structures for your NN,
        and has functions to test the three main aspects of the training.
        The data structures should not change after each step of training.
        You can add to the class, but do not change the
        function headers or return types.
        Make sure that these functions work with a loop to call them multiple times,
        instead of implementing training over multiple steps in the function
    """

    def __init__(self, french_window_size, english_window_size, french_vocab_size, english_vocab_size):
        """
        Initialize a Seq2Seq Model with the given data.

        :param french_window_size: max len of French padded sentence, integer
        :param english_window_size: max len of English padded sentence, integer
        :param french_vocab_size: Vocab size of french, integer
        :param english_vocab_size: Vocab size of english, integer
        """

        # Initialize Placeholders
        self.french_vocab_size, self.english_vocab_size = french_vocab_size, english_vocab_size

        self.encoder_input = tf.placeholder(tf.int32, shape=[None, french_window_size], name='french_input')
        self.encoder_input_length = tf.placeholder(tf.int32, shape=[None], name='french_length')

        self.decoder_input = tf.placeholder(tf.int32, shape=[None, english_window_size], name='english_input')
        self.decoder_input_length = tf.placeholder(tf.int32, shape=[None], name='english_length')
        self.decoder_labels = tf.placeholder(tf.int32, shape=[None, english_window_size], name='english_labels')

        # Please leave these variables
        self.logits = self.forward_pass()
        self.loss = self.loss_function()
        self.train = self.back_propagation()

    def forward_pass(self):
        """
        Calculates the logits

        :return: A tensor of size [batch_size, english_window_size, english_vocab_size]
        """
        EMB_SIZE = 100
        RNN_SIZE = 100
        keep_prob = 0.7
        with tf.variable_scope("enc"):
            F = tf.Variable(tf.random_normal((self.french_vocab_size, EMB_SIZE), stddev = 0.1))
            emb = tf.nn.embedding_lookup(F, self.encoder_input)
            emb = tf.nn.dropout(emb, keep_prob=keep_prob)
            cell = tf.contrib.rnn.LSTMCell(RNN_SIZE)
            init_state = cell.zero_state(BATCH_SIZE, tf.float32)
            enc_out, enc_state = tf.nn.dynamic_rnn(cell, emb, initial_state=init_state )

            attn = tf.Variable(tf.random_normal([FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE], stddev=0.1))
            enc_out_a = tf.tensordot(enc_out, attn, [[1], [0]])

        with tf.variable_scope("dec"):
            E = tf.Variable(tf.random_normal((self.english_vocab_size, EMB_SIZE), stddev=0.1))
            emb = tf.nn.embedding_lookup(E, self.decoder_input)
            emb = tf.nn.dropout(emb, keep_prob=keep_prob)
            cell = tf.contrib.rnn.LSTMCell(RNN_SIZE)

            attn_d = tf.transpose(enc_out_a,[0,2,1])
            emb = tf.concat([emb, attn_d],2)
            dec_out, dec_state = tf.nn.dynamic_rnn(cell, emb, initial_state=enc_state)

        weight1 = tf.Variable(tf.random_normal([RNN_SIZE, self.english_vocab_size], stddev=0.1))
        bias1 = tf.Variable(tf.random_normal([self.english_vocab_size], stddev=0.1))
        logits = tf.tensordot(dec_out, weight1,[[2],[0]]) + bias1

        # weight2 = tf.Variable(tf.random_normal([self.english_vocab_size, self.english_vocab_size], stddev=0.1))
        # bias2 = tf.Variable(tf.random_normal([self.english_vocab_size, self.english_vocab_size], stddev=0.1))
        # logits2 = tf.matmul(logits, weight2) + bias2

        return logits


    def loss_function(self):
        """
        Calculates the model cross-entropy loss after one forward pass

        :return: the loss of the model as a tensor (averaged over batch)
        """
        mask = tf.sequence_mask(self.decoder_input_length+1, ENGLISH_WINDOW_SIZE, dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(self.logits, self.decoder_labels, mask)
        return loss

    def back_propagation(self):
        """
        Adds optimizer to computation graph

        :return: optimizer
        """
        train = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        return train


def main():
    # Load padded corpus
    train_french, train_english, train_french_lengths, train_english_lengths = pad_corpus(TRAIN_FR, TRAIN_EN)
    test_french, test_english, test_french_lengths, test_english_lengths = pad_corpus(TEST_FR, TEST_EN)

    # 1: Build French, English Vocabularies (dictionaries mapping word types to int ids)
    # 2: Creates batches. Remember that the English Decoder labels need to be shifted over by 1.
    # 3. Initialize model
    # 4: Launch Tensorflow Session
    #       -Train
    #       -Test

    with open("french_train.txt", 'rt', encoding = 'latin') as french:
        french_sentences = french.read().split()

    french_vocab = set(french_sentences)

    french_vocab_dict = {w:i for i, w in enumerate(list(french_vocab))}

    french_vocab_dict[STOP_TOKEN] = len(french_vocab_dict)
    # print(french_vocab_dict,"\n", len(french_vocab_dict))
    # exit(0)


    with open("english_train.txt", 'rt', encoding = 'latin') as english:
        english_sentences = english.read().split()

    english_vocab = set(english_sentences)
    english_vocab_dict = {w:i for i,w in enumerate(list(english_vocab))}

    english_vocab_dict[STOP_TOKEN] = len(english_vocab_dict)
    # print(english_vocab_dict, "\n", len(english_vocab_dict))





    french_train_length = len(train_french)
    english_train_length = len(train_english)
    french_test_length = len(test_french)
    english_test_length = len(test_english)

    # print(len(train_french), len(train_french[0]))
    """
    # exit(0)
    i = 0
    j = 0
    while (i < french_train_length):
        while(j < len(train_french[i])):

            var = french_vocab_dict[train_french[i][j]]
            train_french[i][j] = var
            j += 1
        i += 1

    print(train_french[0])

    i = 0
    j = 0
    while (i < english_train_length):
        while(j < len(train_english[i])):
            var = english_vocab_dict[train_english[i][j]]
            train_english[i][j] = var
            j += 1
        i += 1

    print(train_english[0])


    i = 0
    j = 0
    while (i < french_test_length):
        while(j < len(test_french[i])):
            var= french_vocab_dict[test_french[i][j]]
            # print("var: ", var)
            test_french[i][j] =var
            j += 1
        i += 1
    print("sfsdf", test_french[4])

    i = 0
    j = 0
    while (i < english_test_length):
        while(j < len(test_english[i])):
            var = english_vocab_dict[test_english[i][j]]
            test_english[i][j] = var
            j += 1
        i += 1
    """
    # print(test_english[3])



    for i in range(len(train_french)):
        for j in range(len(train_french[i])):
            var =french_vocab_dict[train_french[i][j]]
            train_french[i][j] = var


    for i in range(len(train_english)):
        for j in range(len(train_english[i])):
            var =english_vocab_dict[train_english[i][j]]
            train_english[i][j] = var



    """___________"""
    # print(train_french[0])

    #
    # exit(0)
    english_decoded = []
    french_encoded = []
    french_encoded_lengths = [ ]
    english_decoded_labels = [ ]
    i = 0
    while( i < english_train_length):
        var = train_english[i]
        var = var[:len(var)-1]
        english_decoded.append(var)
        i += 1

    i = 0
    while( i < english_train_length):
        var = train_english[i]
        var = var[1:len(var)]
        english_decoded_labels.append(var)
        i += 1
    # print("1",english_decoded_labels[4])
    #
    # exit(0)

    # batches for french training examples


    num_batches = math.floor((len(train_french) - 1)/ BATCH_SIZE)
    i = 0
    current_batch = []
    current_lengths = []
    while (i < num_batches):
        j = 0
        current_batch = []
        current_lengths = []
        while( j < BATCH_SIZE):
            current_batch.append(train_french[i + BATCH_SIZE * j])
            current_lengths.append(train_french_lengths[i + BATCH_SIZE * j])
            j += 1
        french_encoded.append(current_batch)
        french_encoded_lengths.append(current_lengths)
        i += 1
    
    # batches for english training examples



    """-------2"""



    english_encoded = []
    english_encoded_lengths = []
    num_batches = math.floor((len(english_decoded))/ BATCH_SIZE)
    num_batches_train = num_batches
    i = 0
    while (i < num_batches):
        j = 0
        current_batch = []
        current_lengths = []
        while( j < BATCH_SIZE):
            current_batch.append(english_decoded[i + BATCH_SIZE * j])
            current_lengths.append(train_english_lengths[i + BATCH_SIZE * j])
            j += 1
        english_encoded.append(current_batch)
        english_encoded_lengths.append(current_lengths)
        i += 1
    # print("aa", english_encoded[3])
    # exit(0)

    english_dec_labels = []
    english_label_lengths = []
    num_batches = math.floor((len(english_decoded_labels))/BATCH_SIZE)
    i = 0
    while (i < num_batches):
        j = 0
        current_batch = []
        current_lengths = []
        while( j < BATCH_SIZE):
            current_batch.append(english_decoded_labels[i + BATCH_SIZE * j])
            current_lengths.append(train_english_lengths[i + BATCH_SIZE * j])
            j += 1
        english_dec_labels.append(current_batch)
        english_label_lengths.append(current_lengths)
        i += 1

    # print("2",english_dec_labels[4])


    """----------3"""

    for i in range(len(test_french)):
        for j in range(len(test_french[i])):
            var = french_vocab_dict[test_french[i][j]]
            test_french[i][j] = var

    for i in range(len(test_english)):
        for j in range(len(test_english[i])):
            var = english_vocab_dict[test_english[i][j]]
            test_english[i][j] = var

    english_decoded_t = []
    i = 0
    while(i < english_test_length):
        english_decoded_t.append(test_english[i][: len(test_english[i]) - 1])
        i += 1

    english_decoded_t_labels = list()
    i = 0
    while(i < english_test_length):
        english_decoded_t_labels.append(test_english[i][1: len(test_english[i])])
        i += 1
        
    ####### CHECK LATER ##########
    french_encoded_input_t = list()
    french_encoded_lengths_t = list()
    num_batches = math.floor((french_test_length)/ BATCH_SIZE)
    num_batches_test = num_batches
    i = 0
    while (i < num_batches):
        j = 0
        current_batch = list()
        current_lengths = list()
        while( j < BATCH_SIZE):
            current_batch.append(test_french[i + BATCH_SIZE * j])
            current_lengths.append(test_french_lengths[i + BATCH_SIZE * j])
            j += 1
        french_encoded_input_t.append(current_batch)
        french_encoded_lengths_t.append(current_lengths)
        i += 1
    
    # batches for english training examples

    english_encoded_input_t = list()
    english_encoded_lengths_t = list()
    # num_batches = math.floor((len(english_decoded) - 1)/ BATCH_SIZE)
    i = 0
    while (i < num_batches):
        j = 0
        current_batch = list()
        current_lengths = list()
        while( j < BATCH_SIZE):
            current_batch.append(english_decoded_t[i + BATCH_SIZE * j])
            current_lengths.append(test_english_lengths[i + BATCH_SIZE * j])
            j += 1
        english_encoded_input_t.append(current_batch)
        english_encoded_lengths_t.append(current_lengths)
        i += 1

    english_decoded_labels_t = list()
    english_label_lengths_t = list()
    num_batches = math.floor((len(english_decoded_t_labels) - 1)/ BATCH_SIZE)
    i = 0
    while (i < num_batches):
        j = 0
        current_batch = list()
        current_lengths = list()
        while( j < BATCH_SIZE):
            current_batch.append(english_decoded_t_labels[i + BATCH_SIZE * j])
            current_lengths.append(test_english_lengths[i + BATCH_SIZE * j])
            j += 1
        english_decoded_labels_t.append(current_batch)
        english_label_lengths_t.append(current_lengths)
        i += 1

    
    """------4"""

    model = Model(FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE, french_vocab_size=len(french_vocab_dict),
                  english_vocab_size=len(english_vocab_dict))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for idx in range(num_batches_train):
            # print("inside train")



            l, _= sess.run([model.loss, model.train], feed_dict={model.encoder_input:french_encoded[idx],
                                               model.encoder_input_length:french_encoded_lengths[idx],
                                               model.decoder_input: english_encoded[idx],
                                               model.decoder_input_length:english_encoded_lengths[idx],
                                               model.decoder_labels:english_dec_labels[idx]
                                                   })
            # print(".")
            # print(loss)


        total_loss = 0
        denom = 0
        total_correct_count = 0
        total_word_count = 0
        # print("in test")
        ash = 0
        d = 0
        for idx in range(num_batches_test):

            # print("running")
            loss, y = sess.run([model.loss, model.logits],
                               feed_dict={ model.encoder_input_length: french_encoded_lengths_t[idx],
                                           model.encoder_input: french_encoded_input_t[idx],
                                           model.decoder_input: english_encoded_input_t[idx],
                                           model.decoder_input_length: english_encoded_lengths_t[idx],
                                           model.decoder_labels: english_decoded_labels_t[idx]
                                        })


            # y == logits
            l = loss * (sum(english_encoded_lengths_t[idx]))
            y = np.argmax(y, axis = 2)
            total_loss+=l
            total_words_per_batch = 0
            total_correct_batch = 0
            for p in range(BATCH_SIZE):
                for q in range(ENGLISH_WINDOW_SIZE):
                    total_words_per_batch += 1
                    # print(y[p][q], "\n", english_decoded_labels_t[idx][p][q])
                    if(y[p][q] == english_decoded_labels_t[idx][p][q]):
                        total_correct_batch += 1
            total_correct_count += total_correct_batch
            total_word_count += total_words_per_batch


            denom += sum(english_encoded_lengths_t[idx])

        loss = total_loss/denom
        perplexity = np.exp(loss)
        print("perplexity is :", perplexity)
        # accuracy = total_correct_count/total_word_count
        # print("Accuracy is: ", accuracy)





if __name__ == '__main__':
    main()



"""

Human: What do we want!?
Computer: Natural language processing!
Human: When do we want it!?
Computer: When do we want what?

"""
