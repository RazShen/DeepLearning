import dynet as dy
import numpy as np

prefix_embed_size = 128
suffix_embed_size = 128
word_embed_size = 128
lstm_dim = 64
mlp_dim = 32
char_embed_size = 20
char_lstm_dim = 128


class AEmbeddingBLSTM(object):
    def __init__(self, model, word_to_index, tag_to_index, index_to_tags):
        self.index_to_tags = index_to_tags
        self.word_to_index = word_to_index
        self.tag_to_index = tag_to_index



        # initialize the layers of bilstm
        self.lstm_forward_1 = dy.LSTMBuilder(1, word_embed_size, lstm_dim, model)
        self.lstm_backward_1 = dy.LSTMBuilder(1, word_embed_size, lstm_dim, model)
        self.lstm_forward_2 = dy.LSTMBuilder(1, word_embed_size, lstm_dim, model)
        self.lstm_backward_2 = dy.LSTMBuilder(1, word_embed_size, lstm_dim, model)

        # initialize the mlp on the output of the second blstm
        self.w1 = model.add_parameters((mlp_dim, word_embed_size))
        self.w2 = model.add_parameters((len(tag_to_index), mlp_dim))

        self.E = model.add_lookup_parameters((len(word_to_index), word_embed_size))

    def get_representation(self, w):
        if w in self.word_to_index:
            return self.E[self.word_to_index[w]]
        return self.E[self.word_to_index['UUUNKKK']]

    def get_expressions_from_graph(self, words):
        dy.renew_cg()

        # get the word representation
        word_vectors = [self.get_representation(w) for w in words]

        w1 = dy.parameter(self.w1)
        w2 = dy.parameter(self.w2)

        # get the lstm
        lstm_frw_1 = self.lstm_forward_1.initial_state()
        lstm_bwd_1 = self.lstm_backward_1.initial_state()

        # After getting the word vectors and the lstms,Get the output from the first layer of bilstm
        input_to_lstm_frw_1 = word_vectors
        input_to_lstm_bwd_1 = reversed(word_vectors)
        lstm_frw_1_output = lstm_frw_1.transduce(input_to_lstm_frw_1)
        lstm_bwd_1_output = lstm_bwd_1.transduce(input_to_lstm_bwd_1)
        b_output_from_first_lstm = []

        for exp_from_lstm_frw, exp_from_lstm_bwd in zip(lstm_frw_1_output, lstm_bwd_1_output):
            concat = dy.concatenate([exp_from_lstm_frw, exp_from_lstm_bwd])
            b_output_from_first_lstm.append(concat)

        # after getting the output from the first layer of bilstm, send this output to the second layer of bilstm
        lstm_frw_2 = self.lstm_forward_2.initial_state()
        lstm_bwd_2 = self.lstm_backward_2.initial_state()


        input_to_lstm_frw_2 = b_output_from_first_lstm
        input_to_lstm_bwd_2 = reversed(b_output_from_first_lstm)
        lstm_frw_2_output = lstm_frw_2.transduce(input_to_lstm_frw_2)
        lstm_bwd_2_output = lstm_bwd_2.transduce(input_to_lstm_bwd_2)
        b_tag_output_from_second_lstm = []

        for exp_from_lstm_frw, exp_from_lstm_bwd in zip(lstm_frw_2_output, lstm_bwd_2_output):
            concat = dy.concatenate([exp_from_lstm_frw, exp_from_lstm_bwd])
            b_tag_output_from_second_lstm.append(concat)

        # feed each biLSTM state to an MLP
        output_expression = []
        for x in b_tag_output_from_second_lstm:
            output_expression.append(w2 * (dy.tanh(w1 * x)))

        return output_expression

    def get_loss_on_sentence(self, words, tags):
        """
        Get the total loss on the sentence.
        :param words:
        :param tags:
        :return:
        """
        output_expression = self.get_expressions_from_graph(words)
        losses = []
        for expression, true_tag in zip(output_expression, tags):
            loss = dy.pickneglogsoftmax(expression, self.tag_to_index[true_tag])
            losses.append(loss)
        return dy.esum(losses)

    def get_tags_on_sentence(self, words):
        """
        Get the tags on the sentence by using softmax and get the tag of each word (with the highest probability)
        :param words:
        :return:
        """
        output_as_tags = []
        output_expression = self.get_expressions_from_graph(words)
        output_expression_probabilities = [dy.softmax(v) for v in output_expression]
        # get the value of the expression as numpy array
        output_expression_probabilities = [v.npvalue() for v in output_expression_probabilities]
        for prob in output_expression_probabilities:
            tag = self.index_to_tags[np.argmax(prob)]
            output_as_tags.append(tag)
        return output_as_tags


class BCharBLSTM(AEmbeddingBLSTM):
    def __init__(self, model, word_to_index, tag_to_index, char_to_index, index_to_tags):
        super(BCharBLSTM, self).__init__(model, word_to_index, tag_to_index, index_to_tags)
        self.char_to_index = char_to_index
        self.E_chars = model.add_lookup_parameters((len(char_to_index), char_embed_size))
        self.lstm_for_chars = dy.LSTMBuilder(1, char_embed_size, char_lstm_dim, model) # lstm for the chars

    def get_representation(self, w):
        """
        Get the representation by using the LSTM on the chars of the word.
        :param w:
        :return:
        """
        chars_indexes = [self.char_to_index[c] if c in self.char_to_index else self.char_to_index["UUUNKKK"] for c in w]
        chars_embedding_vectors = [self.E_chars[cid] for cid in chars_indexes]
        char_lstm = self.lstm_for_chars.initial_state()
        output_from_char_lstm = char_lstm.transduce(chars_embedding_vectors)
        return output_from_char_lstm[-1]


# C a_pos_m - embeddings + Affixes
class CEmbeddingPrefSuffBLSTM(AEmbeddingBLSTM):
    def __init__(self, model, word_to_index, tag_to_index, prefix_to_index, suffix_to_index, index_to_tags):
        super(CEmbeddingPrefSuffBLSTM, self).__init__(model, word_to_index, tag_to_index, index_to_tags)
        self.prefix_to_index = prefix_to_index
        self.suffix_to_index = suffix_to_index
        self.prefix_embeddings = model.add_lookup_parameters((len(prefix_to_index), prefix_embed_size))
        self.suffix_embeddings = model.add_lookup_parameters((len(suffix_to_index), suffix_embed_size))

    def get_representation(self, w):
        """
        Get the representation of w by summing the prefix vector embed vector and suffix vector.
        :param w:
        :return:
        """
        final_embeddings = super(CEmbeddingPrefSuffBLSTM, self).get_representation(w)
        if len(w) < 3:
            return final_embeddings
        if w[:3] in self.prefix_to_index:
            prefix_index = self.prefix_to_index[w[:3]]
        else:
            prefix_index = self.prefix_to_index['UUUNKKK'[:3]]
        if w[-3:] in self.suffix_to_index:
            suffix_idx = self.suffix_to_index[w[-3:]]
        else:
            suffix_idx = self.suffix_to_index['UUUNKKK'[-3:]]

        prefix_embeddings = self.prefix_embeddings[prefix_index]
        suffix_embeddings = self.suffix_embeddings[suffix_idx]

        return dy.esum([prefix_embeddings, final_embeddings, suffix_embeddings])


class ABConcatBLSTM(BCharBLSTM):
    """
    D model is a a concatenation of a and b models.
    """

    def __init__(self, model, word_to_index, tag_to_index, char_to_index, index_to_tags):
        super(ABConcatBLSTM, self).__init__(model, word_to_index, tag_to_index, char_to_index, index_to_tags)

        self.w = model.add_parameters((word_embed_size, word_embed_size * 2))
        self.b = model.add_parameters((word_embed_size))

    def get_representation(self, w):
        """
        Get the representation of w by using a and b bilstm output and concatenate it.
        :param w:
        :return:
        """
        a_representation = AEmbeddingBLSTM.get_representation(self, w)
        b_representation = BCharBLSTM.get_representation(self, w)

        final_representation = dy.concatenate([a_representation, b_representation])

        w = dy.parameter(self.w)
        b = dy.parameter(self.b)

        output = w * final_representation
        output += b
        return output
