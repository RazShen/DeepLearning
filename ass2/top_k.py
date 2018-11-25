import utils_part1 as utils
import torch.nn.functional as F
from numpy import linalg as LA
import numpy
import torch

def most_similar(word, k):
    word_embeddings = utils.get_word_embedding("wordVectors.txt", "vocab.txt")
    k_most_similar = []
    print(cosine_distance(word_embeddings[word], word_embeddings["england"]))

def cosine_distance(u,v):
    denominator = numpy.max([float(LA.norm(u, 2) * LA.norm(v, 2)), 1e-8])
    numerator = numpy.dot(u,v)
    return 1.0 * numerator / denominator


word_to_lookup = ["dog", "england", "john", "explode", "office"]
most_similar("dog", 5)