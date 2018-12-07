import rstr
from random import shuffle
from itertools import izip

NUM_EXAMPLES = 500

STUDENT = {'name': 'Raz Shenkman',
           'ID': '311130777'}


def gen_specific_examples(file_name, pos=True):
    """
    generate specific examples for the model.
    :param file_name: to write the examples into
    :param pos: if true generate examples with tag 1
    :return:
    """
    with open(file_name, "w") as file:
        for i in xrange(NUM_EXAMPLES):
            if pos:
                file.write(rstr.xeger(r'[1-9]{1,15}a{1,15}[1-9]{1,15}b{1,15}[1-9]{1,15}'
                                      r'c{1,15}[1-9]{1,15}d{1,15}[1-9]{1,15}') + "\n")
            else:
                file.write(rstr.xeger(r'[1-9]{1,15}a{1,15}[1-9]{1,15}c{1,15}'
                                      r'[1-9]{1,15}b{1,15}[1-9]{1,15}d{1,15}[1-9]{1,15}') + "\n")


def gen_test_and_train(pos_examples, neg_examples):
    """
    split the positive and negative examples into train and test
    :param pos_examples:
    :param neg_examples:
    :return:
    """
    positive_examples = open(pos_examples, "r").readlines()
    negative_examples = open(neg_examples, "r").readlines()
    all_examples = []
    for pos_example, neg_example in izip(positive_examples, negative_examples):
        all_examples.append((pos_example.strip("\n"), 1))
        all_examples.append((neg_example.strip("\n"), 0))
    shuffle(all_examples)
    train_list = all_examples[:int(float(len(all_examples) * 0.8))]
    test_list = all_examples[int(float(len(all_examples) * 0.8)):]
    with open("train", "w") as train:
        for train_example_tuple in train_list:
            train.write(str(train_example_tuple[0]) + " " + str(train_example_tuple[1]) + "\n")
    with open("test", "w") as test:
        for test_example_tuple in test_list:
            test.write(str(test_example_tuple[0]) + " " + str(test_example_tuple[1]) + "\n")


def gen_all_examples():
    """
    generate all of the examples (positive and negative)
    :return:
    """
    gen_specific_examples("pos_examples")
    gen_specific_examples("neg_examples", pos=False)


gen_all_examples()
gen_test_and_train("pos_examples", "neg_examples")
