import rstr
from itertools import izip
NUM_EXAMPLES = 500

def gen_specific_examples(file_name, pos=True):
    with open(file_name,"w") as file:
        for i in xrange(NUM_EXAMPLES):
            if pos:
                file.write(rstr.xeger(r'[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+') + "\n")
            else:
                file.write(rstr.xeger(r'[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+') + "\n")

def gen_test_and_train(pos_examples, neg_examples):
    positive_examples = open(pos_examples, "r").readlines()
    negative_examples = open(neg_examples, "r").readlines()
    with open("train", "w") as train:
        for pos_example, neg_example in izip(positive_examples, negative_examples):
            



def gen_all_examples():
    gen_specific_examples("pos_examples")
    gen_specific_examples("neg_examples", pos=False)

