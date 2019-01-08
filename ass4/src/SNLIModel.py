import abc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


def softmax_attention_3d(values):
    """
    softmax_attention_3d function.
    calculate softmax on the given 3d tensor to get prob vec
    :param values: 3d tensor
    :return: probs 3d vector
    """
    shape = tf.shape(values)
    units = shape[2]
    reshaped_values = tf.reshape(values, tf.stack([-1, units]))
    softmaxes = tf.nn.softmax(reshaped_values)
    return tf.reshape(softmaxes, shape)


def clip_input_sentence(sentence, sizes):
    """
    clip_input_sentence function.
    reshape of the given sentence so that it will have the same length as the longest sentence in the batch.
    :param sentence: input sensence
    :param sizes: batch sentences size
    :return: reshaped sentence
    """
    max_batch_size = tf.reduce_max(sizes)
    clipped_sent = tf.slice(sentence, [0, 0],
                            tf.stack([-1, max_batch_size]))
    return clipped_sent


def get_3d_max_of_martix_batch(values, sentence_sizes, mask_value, dimension=2):
    """
    get_3d_max_of_martix_batch function.
    the function mask the last cols in the attention matrix in case sentence 2 is shorter than
    the maximum.
    :param values: tensor batch_size X m X n
    :param sentence_sizes: sentence sizes to be limited
    :param mask_value: padding val
    :param dimension: dimention from wich we need to operate padding
    :return: tensor
    """
    if dimension == 1:
        values = tf.transpose(values, [0, 2, 1])
    time_steps1 = tf.shape(values)[1]
    time_steps2 = tf.shape(values)[2]

    ones = tf.ones_like(values, dtype=tf.float32)
    pad_values = mask_value * ones
    mask = tf.sequence_mask(sentence_sizes, time_steps2)

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
<<<<<<< HEAD
        constructor.
        creates the model.
        :param num_units: hidden dim.
        :param vocab_size: as its name.
        :param embedding_size: embedding dim.
=======
        Create the model based on MLP networks.

        :param num_units: main dimension of the internal networks
        :param vocab_size: size of the vocabulary
        :param embedding_size: size of each word embedding
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
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
<<<<<<< HEAD
        self.embeddings_ph = tf.placeholder(tf.float32, (vocab_size, embedding_size), 'embeddings')
=======

        # we have to supply the vocab size to allow validate_shape on the
        # embeddings variable, which is necessary down in the graph to determine
        # the shape of inputs at graph construction time
        self.embeddings_ph = tf.placeholder(tf.float32, (vocab_size, embedding_size), 'embeddings')
        # sentence plaholders have shape (batch, time_steps)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        self.sen1 = tf.placeholder(tf.int32, (None, None), 'sentence1')
        self.sen2 = tf.placeholder(tf.int32, (None, None), 'sentence2')
        self.sen1_len = tf.placeholder(tf.int32, [None], 'sent1_size')
        self.sen2_len = tf.placeholder(tf.int32, [None], 'sent2_size')
        self.label = tf.placeholder(tf.int32, [None], 'label')
        self.lr = tf.placeholder(tf.float32, [], name='learning_rate')
        self.l2_c = tf.placeholder(tf.float32, [], 'l2_constant')
        self.clipping_val = tf.placeholder(tf.float32, [], 'clip_norm')
        self.dropout_keep_percentage = tf.placeholder(tf.float32, [], 'dropout')
        self.embed_size = embedding_size
<<<<<<< HEAD
=======
        # we initialize the embeddings from a placeholder to circumvent
        # tensorflow's limitation of 2 GB nodes in the graph
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        """
        Variable allows to keep object as equation, until we create the computation graph.
        """
        self.E = tf.Variable(self.embeddings_ph, trainable=False,
                             validate_shape=True)

        """
        Here self.sentence1 and self.sentence2 are placeholders,
        clip_sentence takes every batch in the input and shortens each
        sentence by the length of the longest sentence in the batch.
        """
        clipped_sen1 = clip_input_sentence(self.sen1, self.sen1_len)
        clipped_sen2 = clip_input_sentence(self.sen2, self.sen2_len)
        embed1 = tf.nn.embedding_lookup(self.E, clipped_sen1)
        embed2 = tf.nn.embedding_lookup(self.E, clipped_sen2)
        rep1 = self.transform_input(embed1)
        rep2 = self.transform_input(embed2, True)
<<<<<<< HEAD
=======
        # the architecture has 3 main steps: soft align, compare and aggregate
        # alpha and beta have shape (batch, time_steps, embeddings)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        self.alpha, self.beta = self.attend_step(rep1, rep2)
        self.V1 = self.compare_with_soft_alignment(rep1, self.beta)
        self.V2 = self.compare_with_soft_alignment(rep2, self.alpha, True)
        self.aggregated_v1_v2 = self.aggregate_v_representations(self.V1, self.V2)
        self.prediction = tf.argmax(self.aggregated_v1_v2, 1, 'answer')
<<<<<<< HEAD
=======

>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        hits = tf.equal(tf.cast(self.prediction, tf.int32), self.label)
        self.acc = tf.reduce_mean(tf.cast(hits, tf.float32),
                                  name='accuracy')
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.aggregated_v1_v2, labels=self.label)
        self.labeled_loss = tf.reduce_mean(cross_entropy)
        w_trust = [v for v in tf.trainable_variables() if 'weight' in v.name]
        l2_part_sum = sum([tf.nn.l2_loss(weight) for weight in w_trust])
        l2_loss = tf.multiply(self.l2_c, l2_part_sum, 'l2_loss')
        self.final_loss = tf.add(self.labeled_loss, l2_loss, 'loss')
        self.create_tensors_for_train()

    def transform_input(self, inputs, reuse_weights=False):
        """
<<<<<<< HEAD
        transform_input function.
        operate transformations on the embeddings.
        :param inputs: a tensor batch X time_steps X embeddings)
        :param reuse_weights: indicates whther to reuse wights.
        :return: tensor
=======
        Apply any transformations to the input embeddings

        :param inputs: a tensor with shape (batch, time_steps, embeddings)
        :param reuse_weights: if to reuse weights or not
        :return: a tensor of the same shape of the input
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        """
        project_embed = self.project_embeddings(inputs, reuse_weights)
        self.representation_size = self.num_units
        if self.use_intra:
<<<<<<< HEAD
=======
            # here, repr's have shape (batch , time_steps, 2*num_units)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
            transformed_embed = self.get_intra_attention_of_sentence(project_embed, reuse_weights)
            self.representation_size = self.representation_size * 2

        return transformed_embed

    def get_distance_biases_for_intra_attendtion(self, time_steps, reuse_weights=False):
        """
        get_distance_biases_for_intra_attendtion function.
        returns tensor of distance biases to be used on the intra-attention matrix.
        :return: tensor
        """
        with tf.variable_scope('distance-bias', reuse=reuse_weights):
            distance_bias_for_intra = tf.get_variable('dist_bias',
                                                      [self.distance_biases], initializer=tf.zeros_initializer())
            range = tf.range(0, time_steps)
            raw_indexes = tf.tile(tf.reshape(range, [1, -1]), tf.stack([time_steps, 1])) - tf.reshape(range, [-1, 1])
            clipped_indexes = tf.clip_by_value(raw_indexes, 0, self.distance_biases - 1)
        return tf.nn.embedding_lookup(distance_bias_for_intra, clipped_indexes)

    def get_intra_attention_of_sentence(self, sentence, reuse_weights=False):
        """
        get_intra_attention_of_sentence fnction.
        calculates the intra attention of a sentence.
        :param sentence: tensor of sentence
        :return: reshaped tensor
        """
        time_steps = tf.shape(sentence)[1]
        with tf.variable_scope('intra-attention') as scope:
<<<<<<< HEAD
            f_intra_of_sen = self.apply_two_feed_forward_layers(sentence, scope, reuse_weights=reuse_weights)
            f_intra_transpose = tf.transpose(f_intra_of_sen, [0, 2, 1])
            with tf.device('/cpu:0'):
                bias = self.get_distance_biases_for_intra_attendtion(time_steps, reuse_weights=reuse_weights)
=======
            # this is F_intra in the paper
            # f_intra1 is (batch, time_steps, num_units) and
            # f_intra1_t is (batch, num_units, time_steps)
            f_intra_of_sen = self.apply_two_feed_forward_layers(sentence, scope, reuse_weights=reuse_weights)
            f_intra_transpose = tf.transpose(f_intra_of_sen, [0, 2, 1])

            # these are f_ij
            # raw_attentions is (batch, time_steps, time_steps)

            # bias has shape (time_steps, time_steps)
            with tf.device('/cpu:0'):
                bias = self.get_distance_biases_for_intra_attendtion(time_steps, reuse_weights=reuse_weights)

            # bias is broadcast along batches
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
            attentions = softmax_attention_3d(tf.matmul(f_intra_of_sen, f_intra_transpose) + bias)
            attended = tf.matmul(attentions, sentence)
        return tf.concat(axis=2, values=[sentence, attended])

    def create_tensors_for_train(self):
        """
        create_tensors_for_train function.
        the function creates tensor objects for training operation.
        """
        with tf.name_scope('training'):
            optim = tf.train.AdagradOptimizer(self.lr)
            grads, v = zip(*optim.compute_gradients(self.final_loss))
            if self.clipping_val is not None:
                grads, _ = tf.clip_by_global_norm(grads, self.clipping_val)
            self.train_optimizer = optim.apply_gradients(zip(grads, v))

    def project_embeddings(self, embeddings, reuse_weights=False):
        """
        project_embeddings function.
        project the word embeddings (the glove dataset)
        :param embeddings: embedded sentence.
        :param reuse_weights: indicates whether to reuse weights.
        :return: the embeddings after being projectes
        """
        time_steps = tf.shape(embeddings)[1]
        """
        When total elements in array is 100 (for example), tf.reshape with -1 makes sure
        that the shape of -1 dimension is adjusting itself to complete to 100 elements in the new shape.
        """
        embeddings_2d = tf.reshape(embeddings, [-1, self.embed_size])
<<<<<<< HEAD
=======

>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        with tf.variable_scope('projection', reuse=reuse_weights):
            initializer = tf.random_normal_initializer(0.0, 0.1)
            projected = tf.matmul(embeddings_2d, tf.get_variable('weights', [self.embed_size, self.num_units],
                                                                 initializer=initializer))
        projected_3d_embed = tf.reshape(projected, tf.stack([-1, time_steps, self.num_units]))
        return projected_3d_embed

    def apply_tranformation_before_comparing(self, sentence, reuse_weights=False):
        """
<<<<<<< HEAD
        apply_tranformation_before_comparing function.
        operates transformation on the attended tokens before the comparing operation.
        :param sentence: sentence represented by tensor.
        :param reuse_weights: indicates whether to reuse weights
        :return: tensor
=======
        Apply the transformation on each attended token before comparing.
        In the original model, it is a two layer feed forward network.

        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        """
        return self.apply_two_feed_forward_layers(sentence, self.compare_scope, reuse_weights)

    def apploy_transformation_before_attending(self, sentence, reuse_weights=False):
        """
<<<<<<< HEAD
        apploy_transformation_before_attending function.
        operates transformation on the sentences before the attending phase.
        :param sentence: sentence represented by tensor
        :param num_units: integer indicates the third dim of sentence
        :param length: originallen of sentence.
        :param reuse_weights: indicates if we are reusing the weights
        :return: tensor
=======
        Apply the transformation on each sentence before attending over each
        other. In the original model, it is a two layer feed forward network.

        :param sentence: a tensor with shape (batch, time_steps, num_units)
        :param num_units: a python int indicating the third dimension of
            sentence
        :param length: real length of the sentence. Not used in this class.
        :param reuse_weights: whether to reuse weights inside this scope
        :return: a tensor with shape (batch, time_steps, num_units)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        """
        return self.apply_two_feed_forward_layers(sentence, self.attend_scope, reuse_weights)

    def apply_two_feed_forward_layers(self, inputs, scope, reuse_weights=False):
        """
<<<<<<< HEAD
        apply_two_feed_forward_layers function.
        operates two feed forward layers.
        :param inputs: tensor
        :param reuse_weights: boolean param - whether to reuse weights
        :param initializer: a tensorflow initializer
        :param num_units: hidden layer dim
        :return: tensor.
=======
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
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        """
        num_units = [self.num_units, self.num_units]
        scope = scope or 'feedforward'
        with tf.variable_scope(scope, reuse=reuse_weights):
            with tf.variable_scope('layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_percentage)
                relus1 = tf.layers.dense(inputs, num_units[0], tf.nn.relu, kernel_initializer=None)
<<<<<<< HEAD
            with tf.variable_scope('layer2'):
                inputs = tf.nn.dropout(relus1, self.dropout_keep_percentage)
                relus2 = tf.layers.dense(inputs, num_units[1], tf.nn.relu, kernel_initializer=None)
=======

            with tf.variable_scope('layer2'):
                inputs = tf.nn.dropout(relus1, self.dropout_keep_percentage)
                relus2 = tf.layers.dense(inputs, num_units[1], tf.nn.relu, kernel_initializer=None)

>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        return relus2

    def aggregate_input(self, v1, v2):
        """
        aggregate_input function.
        operates the aggregate operation as it was explained in the report.
        params and return vals are all tensors.
        """
<<<<<<< HEAD
=======
        # sum over time steps; resulting shape is (batch, num_units)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        v1 = get_3d_max_of_martix_batch(v1, self.sen1_len, 0, 1)
        v2 = get_3d_max_of_martix_batch(v2, self.sen2_len, 0, 1)
        v1_sum = tf.reduce_sum(v1, 1)
        v2_sum = tf.reduce_sum(v2, 1)
        v1_max = tf.reduce_max(v1, 1)
        v2_max = tf.reduce_max(v2, 1)
        concated = tf.concat(axis=1, values=[v1_sum, v2_sum, v1_max, v2_max])
        return concated

    def attend_step(self, sent1, sent2):
        """
        attend_step function.
        operates attending phase as was explained in report.
        params and return vals are all tensors.
        """
        with tf.variable_scope('inter-attention') as self.attend_scope:
<<<<<<< HEAD
=======
            # this is F in the paper
            # repr1 has shape (batch, time_steps, num_units)
            # repr2 has shape (batch, num_units, time_steps)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
            repr1 = self.apploy_transformation_before_attending(sent1)
            repr2 = self.apploy_transformation_before_attending(sent2, True)
            repr2 = tf.transpose(repr2, [0, 2, 1])
            self.raw_attentions = tf.matmul(repr1, repr2)
<<<<<<< HEAD
            masked_vals = get_3d_max_of_martix_batch(self.raw_attentions, self.sen2_len, -np.inf)
            att_sen1 = softmax_attention_3d(masked_vals)
            att_transposed = tf.transpose(self.raw_attentions, [0, 2, 1])
            masked_vals = get_3d_max_of_martix_batch(att_transposed, self.sen1_len, -np.inf)
            att_sen2 = softmax_attention_3d(masked_vals)
=======

            # now get the attention softmaxes
            masked_vals = get_3d_max_of_martix_batch(self.raw_attentions, self.sen2_len, -np.inf)
            att_sen1 = softmax_attention_3d(masked_vals)

            att_transposed = tf.transpose(self.raw_attentions, [0, 2, 1])
            masked_vals = get_3d_max_of_martix_batch(att_transposed, self.sen1_len, -np.inf)
            att_sen2 = softmax_attention_3d(masked_vals)

>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
            self.inter_att1 = att_sen1
            self.inter_att2 = att_sen2
            alpha = tf.matmul(att_sen2, sent1, name='alpha')
            beta = tf.matmul(att_sen1, sent2, name='beta')
<<<<<<< HEAD
=======

>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        return alpha, beta

    def compare_with_soft_alignment(self, sentence, soft_alignment, reuse_weights=False):
        """
        compare_with_soft_alignment function.
        compare the inpute sentence to its soft alignment.

        :param sentence: embedded sentence
        :param soft_alignment: tensor
        :param reuse_weights: indicates if we need to reuse weights.
        :return: tensor
        """
        with tf.variable_scope('comparison', reuse=reuse_weights) as self.compare_scope:
<<<<<<< HEAD
=======
            # sent_and_alignment has shape (batch, time_steps, num_units)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
            sent_and_alignment = tf.concat(axis=2, values=[sentence, soft_alignment, sentence - soft_alignment,
                                                           sentence * soft_alignment])
            output = self.apply_tranformation_before_comparing(sent_and_alignment, reuse_weights)
        return output

    def aggregate_v_representations(self, V1, V2):
        """
<<<<<<< HEAD
        aggregate_v_representations function.
        params are tensors.
=======
        Aggregate the representations induced from both sentences and their
        representations

        :param V1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        """
        inputs = self.aggregate_input(V1, V2)
        with tf.variable_scope('aggregation') as self.aggregate_scope:
            random_init = tf.random_normal_initializer(0.0, 0.1)
            with tf.variable_scope('linear'):
                weights_linear = tf.get_variable('weights', [self.num_units, self.num_classes], initializer=random_init)
                bias_linear = tf.get_variable('bias', [self.num_classes], initializer=tf.zeros_initializer())
            pre_logits = self.apply_two_feed_forward_layers(inputs, self.aggregate_scope)
            logits_over_classes = tf.nn.xw_plus_b(pre_logits, weights_linear, bias_linear)
        return logits_over_classes

    def init_tf_var(self, session, embeddings):
        """
        init_tf_var function.
        initialize the tensor flow variables.
        :param session: tf session
        :param embeddings: glove embeddings
        """
        session.run(tf.global_variables_initializer(), {self.embeddings_ph: embeddings})

    def _create_batch_feed(self, batch_data, learning_rate, dropout_keep,
                           l2, clip_value):
        """
        _create_batch_feed function.
        creates dict to the tf session.
        """
        return {self.sen1: batch_data.sentences1, self.sen2: batch_data.sentences2, self.sen1_len: batch_data.sizes1,
                self.sen2_len: batch_data.sizes2, self.label: batch_data.labels, self.lr: learning_rate,
                self.dropout_keep_percentage: dropout_keep, self.l2_c: l2, self.clipping_val: clip_value}

    def run_on_dev(self, session, feeds_validation):
        """
        run_on_dev function.
        runs the model on dev.
        :return: dev loss and acc
        """
        loss, acc = session.run([self.final_loss, self.acc], feeds_validation)
        return loss, acc

    def train(self, session, train_dataset, valid_dataset):
        """
<<<<<<< HEAD
        train function.
        main loop traines the model and export graphs and accuracy
        :param session: tf session
        :type train_dataset: the train dataset
        :type valid_dataset: the dev dataset
        """
=======
        Train the model
        :param session: tensorflow session
        :type train_dataset: utils.DatasetComparison
        :type valid_dataset: utils.DatasetComparison
        """
        # this tracks the accumulated loss in a minibatch
        # (to take the average later)
>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
        accumulated_loss = 0
        accumulated_accuracy = 0
        accumulated_num_items = 0
        batch_size = 32
        epochs = 10
        best_acc = 0
        learning_rate = 0.05

        batch_counter = 0
        acc_train_dict = {}
        loss_train_dict = {}
        acc_validation_dict = {}
        loss_validation_dict = {}
        dicts_index = 0

        for i in range(epochs):
            train_dataset.shuffle_sentences()
            batch_index = 0
            """
            batch_index indicates the start position of the current batch,
            batch_index2 is the end position of the current batch, taken from our
            train data set.
            """
            while batch_index < train_dataset.num_items:
                batch_index2 = batch_index + batch_size

                batch = train_dataset.get_batch_from_range(batch_index, batch_index2)
                """
                Batch here is of class ComparisonDataSet. already a subset of train_dataset
                """
                feeds_for_valid = self._create_batch_feed(batch, learning_rate,
                                                          0.8, 0.0, 100)
<<<<<<< HEAD
=======

>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
                ops = [self.train_optimizer, self.final_loss, self.acc]
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
<<<<<<< HEAD
                    valid_loss, valid_acc = self.run_on_dev(session,
                                                            feeds_for_valid)
=======

                    valid_loss, valid_acc = self.run_on_dev(session,
                                                            feeds_for_valid)

>>>>>>> e912f61c3e14bafb2450e2145be1ba160bf873e2
                    msg = '%d completed epochs, %d batches' % (i + 1, batch_counter)
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
        """
        plot_graphs function.
        all params are acc and loss dicts on train and dev.
        """
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
