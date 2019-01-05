import abc
import tensorflow as tf
import numpy as np
import utils
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def attention_softmax3d(values):
    """
    Performs a softmax over the attention values.

    :param values: 3d tensor with raw values
    :return: 3d tensor, same shape as input
    """
    original_shape = tf.shape(values)
    num_units = original_shape[2]
    reshaped_values = tf.reshape(values, tf.stack([-1, num_units]))
    softmaxes = tf.nn.softmax(reshaped_values)
    return tf.reshape(softmaxes, original_shape)


def clip_sentence(sentence, sizes):
    """
    Clip the input sentence placeholders to the length of the longest one in the
    batch. This saves processing time.

    :param sentence: tensor with shape (batch, time_steps)
    :param sizes: tensor with shape (batch)
    :return: tensor with shape (batch, time_steps)
    """
    max_batch_size = tf.reduce_max(sizes)
    clipped_sent = tf.slice(sentence, [0, 0],
                            tf.stack([-1, max_batch_size]))
    return clipped_sent


def mask_3d(values, sentence_sizes, mask_value, dimension=2):
    """
    Given a batch of matrices, each with shape m x n, mask the values in each
    row after the positions indicated in sentence_sizes.

    This function is supposed to mask the last columns in the raw attention
    matrix (e_{i, j}) in cases where the sentence2 is smaller than the
    maximum.

    :param values: tensor with shape (batch_size, m, n)
    :param sentence_sizes: tensor with shape (batch_size) containing the
        sentence sizes that should be limited
    :param mask_value: scalar value to assign to items after sentence size
    :param dimension: over which dimension to mask values
    :return: a tensor with the same shape as `values`
    """
    if dimension == 1:
        values = tf.transpose(values, [0, 2, 1])
    time_steps1 = tf.shape(values)[1]
    time_steps2 = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.float32)
    pad_values = mask_value * ones
    mask = tf.sequence_mask(sentence_sizes, time_steps2)

    # mask is (batch_size, sentence2_size). we have to tile it for 3d
    mask3d = tf.expand_dims(mask, 1)
    mask3d = tf.tile(mask3d, (1, time_steps1, 1))

    masked = tf.where(mask3d, values, pad_values)

    if dimension == 1:
        masked = tf.transpose(masked, [0, 2, 1])

    return masked


class SNLIModel(object):

    abc.__metaclass__ = abc.ABCMeta

    def __init__(self, num_units, vocab_size, embedding_size):
        """
        Create the model based on MLP networks.

        :param num_units: main dimension of the internal networks
        :param num_classes: number of possible classes
        :param vocab_size: size of the vocabulary
        :param embedding_size: size of each word embedding
        :param training: whether to create training tensors (optimizer)
        :param project_input: whether to project input embeddings to a
            different dimensionality
        """
        self.use_intra = True
        self.distance_biases = 10
        self.num_units = num_units
        self.num_classes = 3
        self.project_input = True

        """
        Tensorflow placeholder allows us to create our computation graph,
        without needing the data right away. 
        """

        # we have to supply the vocab size to allow validate_shape on the
        # embeddings variable, which is necessary down in the graph to determine
        # the shape of inputs at graph construction time
        self.embeddings_ph = tf.placeholder(tf.float32, (vocab_size, embedding_size),'embeddings')
        # sentence plaholders have shape (batch, time_steps)
        self.sentence1 = tf.placeholder(tf.int32, (None, None), 'sentence1')
        self.sentence2 = tf.placeholder(tf.int32, (None, None), 'sentence2')
        self.sentence1_size = tf.placeholder(tf.int32, [None], 'sent1_size')
        self.sentence2_size = tf.placeholder(tf.int32, [None], 'sent2_size')
        self.label = tf.placeholder(tf.int32, [None], 'label')
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.l2_constant = tf.placeholder(tf.float32, [], 'l2_constant')
        self.clip_value = tf.placeholder(tf.float32, [], 'clip_norm')
        self.dropout_keep = tf.placeholder(tf.float32, [], 'dropout')
        self.embedding_size = embedding_size
        # we initialize the embeddings from a placeholder to circumvent
        # tensorflow's limitation of 2 GB nodes in the graph
        """
        Variable allows to keep object as equation, until we create the computation graph.
        """
        self.embeddings = tf.Variable(self.embeddings_ph, trainable=False,
                                      validate_shape=True)

        # clip the sentences to the length of the longest one in the batch
        # this saves processing time
        """
        Here self.sentence1 and self.sentence2 are placeholders,
        clip_sentence takes every batch in the input and shortens each
        sentence by the length of the longest sentence in the batch.
        """
        clipped_sent1 = clip_sentence(self.sentence1, self.sentence1_size)
        clipped_sent2 = clip_sentence(self.sentence2, self.sentence2_size)
        embedded1 = tf.nn.embedding_lookup(self.embeddings, clipped_sent1)
        embedded2 = tf.nn.embedding_lookup(self.embeddings, clipped_sent2)
        repr1 = self._transformation_input(embedded1)
        repr2 = self._transformation_input(embedded2, True)
        # the architecture has 3 main steps: soft align, compare and aggregate
        # alpha and beta have shape (batch, time_steps, embeddings)
        self.alpha, self.beta = self.attend(repr1, repr2)
        self.v1 = self.compare(repr1, self.beta, self.sentence1_size)
        self.v2 = self.compare(repr2, self.alpha, self.sentence2_size, True)
        self.logits = self.aggregate(self.v1, self.v2)
        self.answer = tf.argmax(self.logits, 1, 'answer')

        hits = tf.equal(tf.cast(self.answer, tf.int32), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(hits, tf.float32),
                                       name='accuracy')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
        self.labeled_loss = tf.reduce_mean(cross_entropy)
        weights_trust = [v for v in tf.trainable_variables() if 'weight' in v.name]
        l2_partial_sum = sum([tf.nn.l2_loss(weight) for weight in weights_trust])
        l2_loss = tf.multiply(self.l2_constant, l2_partial_sum, 'l2_loss')
        self.loss = tf.add(self.labeled_loss, l2_loss, 'loss')
        self._create_training_tensors()

    def _transformation_input(self, inputs, reuse_weights=False):
        """
        Apply any transformations to the input embeddings

        :param inputs: a tensor with shape (batch, time_steps, embeddings)
        :return: a tensor of the same shape of the input
        """
        projected = self.project_embeddings(inputs, reuse_weights)
        self.representation_size = self.num_units
        if self.use_intra:
            # here, repr's have shape (batch , time_steps, 2*num_units)
            transformed = self.compute_intra_attention(projected,
                                                       reuse_weights)
            self.representation_size *= 2

        return transformed

    def _get_distance_biases(self, time_steps, reuse_weights=False):
        """
        Return a 2-d tensor with the values of the distance biases to be applied
        on the intra-attention matrix of size sentence_size

        :param time_steps: tensor scalar
        :return: 2-d tensor (time_steps, time_steps)
        """
        with tf.variable_scope('distance-bias', reuse=reuse_weights):
            # this is d_{i-j}
            distance_bias = tf.get_variable('dist_bias', [self.distance_biases],
                                            initializer=tf.zeros_initializer())

            # messy tensor manipulation for indexing the biases
            r = tf.range(0, time_steps)
            r_matrix = tf.tile(tf.reshape(r, [1, -1]),
                               tf.stack([time_steps, 1]))
            raw_inds = r_matrix - tf.reshape(r, [-1, 1])
            clipped_inds = tf.clip_by_value(raw_inds, 0,
                                            self.distance_biases - 1)
            values = tf.nn.embedding_lookup(distance_bias, clipped_inds)

        return values

    def compute_intra_attention(self, sentence, reuse_weights=False):
        """
        Compute the intra attention of a sentence. It returns a concatenation
        of the original sentence with its attended output.

        :param sentence: tensor in shape (batch, time_steps, num_units)
        :return: a tensor in shape (batch, time_steps, 2*num_units)
        """
        time_steps = tf.shape(sentence)[1]
        with tf.variable_scope('intra-attention') as scope:
            # this is F_intra in the paper
            # f_intra1 is (batch, time_steps, num_units) and
            # f_intra1_t is (batch, num_units, time_steps)
            f_intra = self._apply_feedforward(sentence, scope,
                                              reuse_weights=reuse_weights)
            f_intra_t = tf.transpose(f_intra, [0, 2, 1])

            # these are f_ij
            # raw_attentions is (batch, time_steps, time_steps)
            raw_attentions = tf.matmul(f_intra, f_intra_t)

            # bias has shape (time_steps, time_steps)
            with tf.device('/cpu:0'):
                bias = self._get_distance_biases(time_steps,
                                                 reuse_weights=reuse_weights)

            # bias is broadcast along batches
            raw_attentions += bias
            attentions = attention_softmax3d(raw_attentions)
            attended = tf.matmul(attentions, sentence)

        return tf.concat(axis=2, values=[sentence, attended])

    def _create_training_tensors(self):
        """
        Create the tensors used for training
        """
        with tf.name_scope('training'):
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss))
            if self.clip_value is not None:
                gradients, _ = tf.clip_by_global_norm(gradients,
                                                      self.clip_value)
            self.train_optimizer = optimizer.apply_gradients(zip(gradients, v))

    def project_embeddings(self, embeddings, reuse_weights=False):
        """
        Project word embeddings into another dimensionality

        :param embeddings: embedded sentence, shape (batch, time_steps,
            embedding_size)
        :param reuse_weights: reuse weights in internal layers
        :return: projected embeddings with shape (batch, time_steps, num_units)
        """
        time_steps = tf.shape(embeddings)[1]
        """
        When total elements in array is 100 (for example), tf.reshape with -1 makes sure
        that the shape of -1 dimension is adjusting itself to complete to 100 elements in the new shape.
        """
        embeddings_2d = tf.reshape(embeddings, [-1, self.embedding_size])

        with tf.variable_scope('projection', reuse=reuse_weights):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            weights = tf.get_variable('weights',
                                      [self.embedding_size, self.num_units],
                                      initializer=initializer)

            projected = tf.matmul(embeddings_2d, weights)

        projected_3d = tf.reshape(projected,
                                  tf.stack([-1, time_steps, self.num_units]))
        return projected_3d

    def _transformation_compare(self, sentence, num_units, length,
                                reuse_weights=False):
        """
        Apply the transformation on each attended token before comparing.
        In the original model, it is a two layer feed forward network.

        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param num_units: a python int indicating the third dimension of
            sentence
        :param length: real length of the sentence. Not used in this class.
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        return self._apply_feedforward(sentence, self.compare_scope,
                                       reuse_weights)

    def _transformation_attend(self, sentence, num_units, length,
                               reuse_weights=False):
        """
        Apply the transformation on each sentence before attending over each
        other. In the original model, it is a two layer feed forward network.

        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param num_units: a python int indicating the third dimension of
            sentence
        :param length: real length of the sentence. Not used in this class.
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        return self._apply_feedforward(sentence, self.attend_scope,
                                       reuse_weights)

    def _apply_feedforward(self, inputs, scope,
                           reuse_weights=False, initializer=None,
                           num_units=None):
        """
        Apply two feed forward layers with self.num_units on the inputs.
        :param inputs: tensor in shape (batch, time_steps, num_input_units)
            or (batch, num_units)
        :param reuse_weights: reuse the weights inside the same tensorflow
            variable scope
        :param initializer: tensorflow initializer; by default a normal
            distribution
        :param num_units: list of length 2 containing the number of units to be
            used in each layer
        :return: a tensor with shape (batch, time_steps, num_units)
        """
        if num_units is None:
            num_units = [self.num_units, self.num_units]

        scope = scope or 'feedforward'
        with tf.variable_scope(scope, reuse=reuse_weights):
            with tf.variable_scope('layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep)
                relus = tf.layers.dense(inputs, num_units[0], tf.nn.relu,
                                        kernel_initializer=initializer)

            with tf.variable_scope('layer2'):
                inputs = tf.nn.dropout(relus, self.dropout_keep)
                relus2 = tf.layers.dense(inputs, num_units[1], tf.nn.relu,
                                         kernel_initializer=initializer)

        return relus2

    def _create_aggregate_input(self, v1, v2):
        """
        Create and return the input to the aggregate step.

        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: a tensor with shape (batch, num_aggregate_inputs)
        """
        # sum over time steps; resulting shape is (batch, num_units)
        v1 = mask_3d(v1, self.sentence1_size, 0, 1)
        v2 = mask_3d(v2, self.sentence2_size, 0, 1)
        v1_sum = tf.reduce_sum(v1, 1)
        v2_sum = tf.reduce_sum(v2, 1)
        v1_max = tf.reduce_max(v1, 1)
        v2_max = tf.reduce_max(v2, 1)

        return tf.concat(axis=1, values=[v1_sum, v2_sum, v1_max, v2_max])

    def attend(self, sent1, sent2):
        """
        Compute inter-sentence attention. This is step 1 (attend) in the paper

        :param sent1: tensor in shape (batch, time_steps, num_units),
            the projected sentence 1
        :param sent2: tensor in shape (batch, time_steps, num_units)
        :return: a tuple of 3-d tensors, alfa and beta.
        """
        with tf.variable_scope('inter-attention') as self.attend_scope:
            # this is F in the paper
            num_units = self.representation_size

            # repr1 has shape (batch, time_steps, num_units)
            # repr2 has shape (batch, num_units, time_steps)
            repr1 = self._transformation_attend(sent1, num_units,
                                                self.sentence1_size)
            repr2 = self._transformation_attend(sent2, num_units,
                                                self.sentence2_size, True)
            repr2 = tf.transpose(repr2, [0, 2, 1])

            # compute the unnormalized attention for all word pairs
            # raw_attentions has shape (batch, time_steps1, time_steps2)
            self.raw_attentions = tf.matmul(repr1, repr2)

            # now get the attention softmaxes
            masked = mask_3d(self.raw_attentions, self.sentence2_size, -np.inf)
            att_sent1 = attention_softmax3d(masked)

            att_transposed = tf.transpose(self.raw_attentions, [0, 2, 1])
            masked = mask_3d(att_transposed, self.sentence1_size, -np.inf)
            att_sent2 = attention_softmax3d(masked)

            self.inter_att1 = att_sent1
            self.inter_att2 = att_sent2
            alpha = tf.matmul(att_sent2, sent1, name='alpha')
            beta = tf.matmul(att_sent1, sent2, name='beta')

        return alpha, beta

    def compare(self, sentence, soft_alignment, sentence_length,
                reuse_weights=False):
        """
        Apply a feed forward network to compare one sentence to its
        soft alignment with the other.

        :param sentence: embedded and projected sentence,
            shape (batch, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch, time_steps, num_units)
        :param reuse_weights: whether to reuse weights in the internal layers
        :return: a tensor (batch, time_steps, num_units)
        """
        with tf.variable_scope('comparison', reuse=reuse_weights) \
                as self.compare_scope:
            num_units = 2 * self.representation_size

            # sent_and_alignment has shape (batch, time_steps, num_units)
            inputs = [sentence, soft_alignment, sentence - soft_alignment,
                      sentence * soft_alignment]
            sent_and_alignment = tf.concat(axis=2, values=inputs)

            output = self._transformation_compare(sent_and_alignment, num_units, sentence_length, reuse_weights)

        return output

    def aggregate(self, v1, v2):
        """
        Aggregate the representations induced from both sentences and their
        representations

        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        inputs = self._create_aggregate_input(v1, v2)
        with tf.variable_scope('aggregation') as self.aggregate_scope:
            initializer = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('linear'):
                shape = [self.num_units, self.num_classes]
                weights_linear = tf.get_variable('weights', shape,
                                                 initializer=initializer)
                bias_linear = tf.get_variable('bias', [self.num_classes],
                                              initializer=tf.zeros_initializer())

            pre_logits = self._apply_feedforward(inputs,
                                                 self.aggregate_scope)
            logits = tf.nn.xw_plus_b(pre_logits, weights_linear, bias_linear)

        return logits

    def initialize_embeddings(self, session, embeddings):
        """
        Initialize word embeddings
        :param session: tensorflow session
        :param embeddings: the contents of the word embeddings
        :return:
        """
        init_op = tf.variables_initializer([self.embeddings])
        session.run(init_op, {self.embeddings_ph: embeddings})

    def initialize(self, session, embeddings):
        """
        Initialize all tensorflow variables.
        :param session: tensorflow session
        :param embeddings: the contents of the word embeddings
        """
        init_op = tf.global_variables_initializer()
        session.run(init_op, {self.embeddings_ph: embeddings})

    def _create_batch_feed(self, batch_data, learning_rate, dropout_keep,
                           l2, clip_value):
        """
        Create a feed dictionary to be given to the tensorflow session.
        """
        feeds = {self.sentence1: batch_data.sentences1,
                 self.sentence2: batch_data.sentences2,
                 self.sentence1_size: batch_data.sizes1,
                 self.sentence2_size: batch_data.sizes2,
                 self.label: batch_data.labels,
                 self.learning_rate: learning_rate,
                 self.dropout_keep: dropout_keep,
                 self.l2_constant: l2,
                 self.clip_value: clip_value
                 }
        return feeds

    def _run_on_validation(self, session, feeds_validation):
        """
        Run the model with validation data, providing any useful information.

        :return: a tuple (validation_loss, validation_acc)
        """
        loss, acc = session.run([self.loss, self.accuracy], feeds_validation)
        return loss, acc

    def train(self, session, train_dataset, valid_dataset):
        """
        Train the model
        :param session: tensorflow session
        :type train_dataset: utils.RTEDataset
        :type valid_dataset: utils.RTEDataset
        :param save_dir: path to a directory to save model files
        :param learning_rate: the initial learning rate
        :param num_epochs: how many epochs to train the model for
        :param batch_size: size of each batch
        :param dropout_keep: probability of keeping units during the dropout
        :param l2: l2 loss coefficient
        :param clip_norm: global norm to clip gradient tensors
        :param report_interval: number of batches before each performance
            report
        """
        # this tracks the accumulated loss in a minibatch
        # (to take the average later)
        accumulated_loss = 0
        accumulated_accuracy = 0
        accumulated_num_items = 0
        batch_size = 32
        epochs = 10
        best_acc = 0
        learning_rate = 0.05

        # batch counter doesn't reset after each epoch
        batch_counter = 0
        acc_train_dict = {}
        loss_train_dict = {}
        acc_validation_dict = {}
        loss_validation_dict = {}
        dicts_index = 0


        for i in range(epochs):
            train_dataset.shuffle_data()
            batch_index = 0
            """
            batch_index indicates the start position of the current batch,
            batch_index2 is the end position of the current batch, taken from our
            train data set.
            """
            while batch_index < train_dataset.num_items:
                batch_index2 = batch_index + batch_size

                batch = train_dataset.get_batch(batch_index, batch_index2)
                """
                Batch here is of class RTEDataset. already a subset of train_dataset
                """
                feeds_for_valid = self._create_batch_feed(batch, learning_rate,
                                                          0.8, 0.0, 100)

                ops = [self.train_optimizer, self.loss, self.accuracy]
                _, loss, accuracy = session.run(ops, feed_dict=feeds_for_valid)
                accumulated_loss += loss * batch.num_items
                accumulated_accuracy += accuracy * batch.num_items
                accumulated_num_items += batch.num_items

                batch_index = batch_index2
                batch_counter += 1

                if batch_counter % 100 == 0:
                    """
                    avg_loss is the loss of all the batches seen so far.
                    """
                    avg_loss = accumulated_loss / accumulated_num_items
                    avg_accuracy = accumulated_accuracy / accumulated_num_items
                    accumulated_loss = 0
                    accumulated_accuracy = 0
                    accumulated_num_items = 0

                    feeds_for_valid = self._create_batch_feed(valid_dataset,
                                                              0, 1, 0.0, 0)

                    valid_loss, valid_acc = self._run_on_validation(session,
                                                                    feeds_for_valid)

                    msg = '%d completed epochs, %d batches' % (i+1, batch_counter)
                    msg += '\tTrain loss: %.4f' % avg_loss
                    msg += '\tTrain acc: %.4f' % (avg_accuracy * 100)
                    msg += '\tTest loss: %.4f' % valid_loss
                    msg += '\tTest acc: %.4f' % (valid_acc * 100)

                    if valid_acc > best_acc:
                        best_acc = valid_acc
                    print(msg)

                    acc_train_dict[dicts_index] = avg_accuracy
                    loss_train_dict[dicts_index] = avg_loss
                    acc_validation_dict[dicts_index] = valid_acc
                    loss_validation_dict[dicts_index] = valid_loss
                    dicts_index += 1

            self.plot_graphs(acc_train_dict, loss_train_dict, acc_validation_dict, loss_validation_dict)

    def plot_graphs(self, acc_train_dict, loss_train_dict, acc_validation_dict, loss_validation_dict):
        # accuracy graph
        label1, = plt.plot(acc_train_dict.keys(), acc_train_dict.values(), "b-", label='Train Avg. Accuracy')
        label2, = plt.plot(acc_validation_dict.keys(), acc_validation_dict.values(), "r-",
                           label='Test Avg. Accuracy')
        plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
        plt.show()

        # loss graph
        label1, = plt.plot(loss_train_dict.keys(), loss_train_dict.values(), "b-", label='Train Avg. Loss')
        label2, = plt.plot(loss_validation_dict.keys(), loss_validation_dict.values(), "r-",
                           label='Test Avg. Loss')
        plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
        plt.show()
