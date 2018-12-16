import pickle
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import collections

def load_dicts_from_modelFile(pkl_name):
    dict = {}
    with open(pkl_name) as dicts_file:
        dict = pickle.load(dicts_file)
    for key in dict:
        tmp = dict[key]
        dict[key] = tmp * 100
    return collections.OrderedDict(sorted(dict.items()))
#
a_pos = load_dicts_from_modelFile("a_graph_pos")
b_pos = load_dicts_from_modelFile("b_graph_pos")
c_pos = load_dicts_from_modelFile("c_graph_pos")
d_pos = load_dicts_from_modelFile("d_graph_pos")

# a_ner = load_dicts_from_modelFile("a_graph_ner")
# b_ner = load_dicts_from_modelFile("b_graph_ner")
# c_ner = load_dicts_from_modelFile("c_graph_ner")
# d_ner = load_dicts_from_modelFile("d_graph_ner")

label1, = plt.plot(a_pos.keys(), a_pos.values(), "b-", label='model a pos')
label2, = plt.plot(b_pos.keys(), b_pos.values(), "r-", label='model b pos')
label3, = plt.plot(c_pos.keys(), c_pos.values(), "g-", label='model c pos')
label4, = plt.plot(d_pos.keys(), d_pos.values(), "y-", label='model d pos')
plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
plt.show()


# label1, = plt.plot(a_ner.keys(), a_ner.values(), "b-", label='model a ner')
# label2, = plt.plot(b_ner.keys(), b_ner.values(), "r-", label='model b ner')
# label3, = plt.plot(c_ner.keys(), c_ner.values(), "g-", label='model c ner')
# label4, = plt.plot(d_ner.keys(), d_ner.values(), "y-", label='model d ner')
# plt.legend(handler_map={label1: HandlerLine2D(numpoints=4)})
# plt.show()
