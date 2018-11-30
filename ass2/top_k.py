import utils_part1 as utils
from numpy import linalg as LA
import numpy

STUDENT = {'name': 'Raz Shenkman',
           'ID': '311130777'}


def most_similar(word, k, get_dist=False):
    """
    Return k most similar words to the word.
    :param word: tested word
    :param k:
    :param get_dist: if to return distances to the words with the similar words, or just the similar words
    :return:
    """
    word_embeddings = utils.get_word_embedding("wordVectors.txt", "vocab.txt")
    distances = []
    for some_word in word_embeddings:
        if some_word == word:
            continue  # skip the word if it is the tested word (would be closest to itself)
        distances.append([some_word, cosine_distance(word_embeddings[word], word_embeddings[some_word])])
    distances = sorted(distances, key=get_distance)
    top_k = distances[-k:]
    if get_dist:
        return sorted(top_k, key=get_distance, reverse=True)
    else:
        return [item[0] for item in sorted(top_k, key=get_distance, reverse=True)]


def get_distance(t):
    """
    Return the distance to the tested word from the element
    :param t: list of 2 elements, first is word, second is cosine distance to tested word
    :return: the distance
    """
    return t[1]


def cosine_distance(u, v):
    """
    returns the cosine distance between u and v
    :param u: word
    :param v: word
    :return: cosine distance
    """
    denominator = numpy.max([float(LA.norm(u, 2) * LA.norm(v, 2)), 1e-8])
    numerator = numpy.dot(u, v)
    return 1.0 * numerator / denominator


word_to_lookup = ["dog", "england", "john", "explode", "office"]
words_similar = []
for word in word_to_lookup:
    print("The most similar words to " + word + " are:")
    print(most_similar(word, 5, True))
    print("\n")
