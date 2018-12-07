def read_examples_file(train_file_path):
    train_as_tuples_of_example_and_tag = []
    with open(train_file_path, "r") as train_file:
        for line in train_file:
            example, tag = line.strip().split()
            train_as_tuples_of_example_and_tag.append((example, int(tag)))
    return train_as_tuples_of_example_and_tag
