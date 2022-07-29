from functools import reduce

from Datasets import fp_clean_dataset
from Node import *


# Reads the dataset and returns it as an array
def read_dataset(filepath):
    dataset = np.loadtxt(filepath, dtype='float')
    dataset_array = np.array(dataset)
    return dataset_array


# Returns the number of attributes of the dataset, 7 in our case
def get_number_attributes(dataset):
    return dataset.shape[1] - 1


# Returns the labels for classification
def get_labels():
    return [1, 2, 3, 4]


# Trains the model based on the dataset provided as argument
# Precondition: attributes are from 0 to 6
def decision_tree_learning(dataset, depth):
    # Base case for empty dataset
    if dataset.size == 0:
        return Leaf(None, depth), depth

    # We reached a leaf (for the current dataset all instances are classified with the same label)
    if all_same_label(dataset):
        return Leaf(dataset[0][-1], depth, dataset.shape[0]), depth

    # Otherwise we find the split value which will be used for partitioning
    (value, attribute) = find_split(dataset)

    # Partition dataset based on split value:
    (left_dataset, right_dataset) = partition_dataset(dataset, value, attribute)

    # Construct left and right subtrees recursively:
    (left_node, left_depth) = decision_tree_learning(left_dataset, depth + 1)
    (right_node, right_depth) = decision_tree_learning(right_dataset, depth + 1)

    # Current depth is max depth of left and right subtrees
    node_depth = max(left_depth, right_depth)

    # Create the current node with the left and right subtrees and return it
    curr_node = Node(value, left_node, right_node, attribute, node_depth)

    return curr_node, node_depth


# Finds the split point as (value, attribute_idx)
# We will use it when checking if the attribute is > or <= than the value
def find_split(dataset):
    # We will keep track of the (value, max_info_gain, attribute_idx)
    max_value_attribute = 0, 0, 0

    for attribute in range(get_number_attributes(dataset)):
        dataset_attributes = dataset[:, attribute]
        all_same_entries = np.unique(dataset_attributes).size == 1

        if not all_same_entries:

            # Calculate the split values for the current attribute
            values = find_split_values(dataset_attributes)

            # Get the left and right partitions for all split values we calculated
            partitions = list(map(lambda x: partition_dataset(dataset, x, attribute), values))

            # Get the corresponding information gain for each partition of (left, right)
            information_gains = map(
                lambda partition: information_gain(dataset, partition[0], partition[1]),
                partitions
            )
            # Map the information gains to the corresponding split value i.e. [(value, gain)]
            value_info_gains = np.array(list(zip(values, information_gains)))

            # Get the maximum info gain calculated
            max_value_info_gain = reduce(lambda vg1, vg2: vg1 if (vg1[1] > vg2[1]) else vg2, value_info_gains)

            # If we found a new maximum info gain we update max_value_attribute
            if max_value_info_gain[1] > max_value_attribute[1]:
                max_value_attribute = max_value_info_gain[0], max_value_info_gain[1], attribute

    # Return (value, attribute_idx)
    return max_value_attribute[0], max_value_attribute[2]


# Computes and returns the median of every 2 consecutive attribute values, representing the split values
def find_split_values(attribute_values):
    # We use np.unique to ensure that for equal consecutive values we have only 1 split point considered
    attribute_values = np.unique(np.sort(attribute_values))
    pairs = np.array([attribute_values[1:], attribute_values[:-1]])

    return np.median(pairs, axis=0)


# Partitions the instances from the given dataset into (left dataset points, right dataset points)
# Left dataset = instances from given dataset where instance[attribute_idx] < value
# Right dataset = instances from given dataset where instance[attribute_idx] >= value
def partition_dataset(dataset, value, attribute_idx):
    left_dataset = []
    right_dataset = []

    # For every instance, if the value of the attribute is < value we append to left list, otherwise we append to right
    for instance in dataset:
        curr_value = instance[attribute_idx]

        if curr_value < value:
            left_dataset.append(instance)
        else:
            right_dataset.append(instance)

    left_dataset = np.array(left_dataset)
    right_dataset = np.array(right_dataset)

    return left_dataset, right_dataset


# Checks if all samples have the same label
def all_same_label(dataset):
    labels = dataset[:, -1]
    return np.unique(labels).size == 1


# Entropy is calculated based on the provided formula: H(X) = - sum(k in K) of (P(xk) * log2(P(xk)))
def entropy(dataset):
    labels = dataset[:, -1]

    # We need the frequency of each lable in the dataset
    _, frequency = np.unique(labels, return_counts=True)

    # Probability function defined as (number of specific label / total number of labels)
    prob_func = lambda x: x / labels.size
    probabilities = prob_func(frequency)

    # Function which calculates P(xk) * log2(P(xk))
    prod_log2 = lambda p: p * np.log2(p)

    # We calculate and return entropy based on the formula
    return (-1) * np.sum(list(map(prod_log2, probabilities)))


# Remainder is calculated by: rem = (|left| * H(left) + |right| * H(right)) / (|left| + |right|)
def remainder(left_dataset, right_dataset):
    # We get the total number of labels of the left and right subtrees
    left_labels_nr = left_dataset[:, -1].size
    right_labels_nr = right_dataset[:, -1].size

    # Calculate |left| * H(left) and |right| * H(right)
    rem_left = left_labels_nr * entropy(left_dataset)
    rem_right = right_labels_nr * entropy(right_dataset)

    # Compute and return the remainder based on the formula
    return (rem_left + rem_right) / (left_labels_nr + right_labels_nr)


# We calculate the information gain by the formula: gain = H(dataset) - remainder(left, right)
def information_gain(dataset, left_dataset, right_dataset):
    # Get H(dataset) = entropy
    dataset_entropy = entropy(dataset)

    # Calculate remainder(left, right)
    rm = remainder(left_dataset, right_dataset)

    return dataset_entropy - rm


# Returns (label, instances): the class label with the maximum instances
# And the new array of instances
def get_majority_class_label(left_subtree, right_subtree):
    total_instances = left_subtree.instances + right_subtree.instances
    result = np.where(total_instances == np.amax(total_instances))

    return result[0][0] + 1, total_instances