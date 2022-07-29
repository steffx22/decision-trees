import numpy as np

from Datasets import fp_dummy_example
from Datasets import fp_clean_dataset
from Datasets import fp_noisy_dataset
from DecisionTree import *
from Visualisation import visualise
from enum import Enum


class Side(Enum):
    LEFT = 1
    RIGHT = 2


class Pruning(Enum):
    WITH_PRUNING = 1
    WITHOUT_PRUNING = 2


# performs pruning on the decision tree until accuracy can no longer be improved
def prune(root, parent, side, tree, validation_dataset):
    if isinstance(tree, Leaf):  # Base case, we can't prune on a leaf
        return tree

    # Recursively prune on left and right subtrees
    tree.left = prune(root, tree, Side.LEFT, tree.left, validation_dataset)
    tree.right = prune(root, tree, Side.RIGHT, tree.right, validation_dataset)

    # If both left and right subtrees are leaves, we have a possibility of pruning
    if isinstance(tree.left, Leaf) and isinstance(tree.right, Leaf):
        # Get the label corresponding to the greatest number of instances
        majority_class, total_instances = get_majority_class_label(tree.left, tree.right)

        # Create a new leaf which would replace the node if we indeed get greater accuracy with it
        new_leaf = Leaf(majority_class, tree.depth, 0)
        new_leaf.instances = total_instances

        # Get accuuracy for the validation_dataset on the current decision tree, i.e. before pruning
        conf_matrix = generate_conf_matrix_from_decision_tree(validation_dataset, root)
        accuracy = get_accuracy(conf_matrix)

        # We try to prune and see if we get bigger accuracy:
        # If parent doesn't exist then current node is the root, which becomes the new leaf
        if parent is None:
            root = new_leaf
        # We're on the left node of our parent, must update parent.left
        elif side == Side.LEFT:
            parent.left = new_leaf
        # We're on the right node of our parent, must update parent.right
        else:
            parent.right = new_leaf

        # Get accuracy for the pruned decision tree
        conf_matrix_pruned = generate_conf_matrix_from_decision_tree(validation_dataset, root)
        accuracy_pruned = get_accuracy(conf_matrix_pruned)

        # If we get bigger accuracy then just return the leaf because we already updated the tree
        if accuracy_pruned >= accuracy:
            return new_leaf

        # Otherwise we must roll back our changes to keep the not pruned decision tree
        else:
            if parent is None:
                root = tree
            elif side == Side.LEFT:
                parent.left = tree
            else:
                parent.right = tree
            return tree
    else:
        # This is the case when we can't prune since left and right are not leaves so we just return the tree
        return tree


# Generates the confusion matrix based on the predicted labels and the correct labels
def generate_confusion_matrix(test_labels, predicted_labels):
    all_labels = get_labels()
    no_labels = len(all_labels)

    # We construct the confusion matrix as 4 by 4 (we have 4 labels in total)
    confusion = np.zeros((no_labels, no_labels))

    # For each label
    for (label_index, label) in enumerate(all_labels):
        # Get predictions for which we have true for that index in indices
        # i.e. the corresponding predictions for the tests that have test_label = label
        indices = (test_labels == label)
        predictions = predicted_labels[indices]

        # Get the count of each label predicted
        (unique_labels, counts) = np.unique(predictions, return_counts=True)
        frequency_dict = dict(zip(unique_labels, counts))

        # Fill up the confusion matrix for the current row, i.e. for the current label
        for (other_label_index, other_label) in enumerate(all_labels):
            confusion[label_index, other_label_index] = frequency_dict.get(other_label, 0)

    return confusion


# Splits the dataset into n_splits folds
def k_fold_split(n_splits, dataset, random_generator=np.random.default_rng()):
    # We shuffle the indices to improve the presence of all 4 labels in the splits
    shuffled_indices = random_generator.permutation(dataset.shape[0])

    # Split the indices into n folds and return them
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def get_depth(tree):
    if isinstance(tree, Leaf):
        return 1
    return 1 + max(get_depth(tree.left), get_depth(tree.right))


# Returns (precision, recall, f1, accuracy) as the accuracy and the mean of precisions, recalls, f1 of the 4 classes
# This method doesn't use a validation dataset and is used for Step 3 only
def get_cross_validation_metrics(k, dataset):
    depth_sum = 0
    k = min(k, dataset.shape[0])

    # Get the k folds
    split_indices = k_fold_split(k, dataset)
    matrices = []

    # We consider each fold of the k ones as test dataset
    # We have one fold of the test_indices and the rest of them i.e. (k - 1) for the training
    for i in range(k):
        # Get the test indices for one fold and the corresponding dataset instances for them
        test_indices = split_indices[i]
        test_dataset = dataset[test_indices, :]

        # Get the rest of (k - 1) folds for training and the corresponding dataset instances for them
        train_indices = np.hstack(split_indices[:i] + split_indices[i + 1:])
        train_dataset = dataset[train_indices, :]

        # Train the tree on the train dataset
        (trained_tree, _) = decision_tree_learning(train_dataset, 0)
        depth_sum += get_depth(trained_tree)

        # Compute the test_labels and predicted_labels for the trained tree on the test dataset
        (test_labels, predicted_labels) = evaluate(test_dataset, trained_tree)

        # Compute the confusion matrix by comparing test_labels with predicted_labels and append it to the 'matrices'
        conf_matrix = generate_confusion_matrix(test_labels, predicted_labels)
        matrices.append(conf_matrix)

    # Calculate total_confusion_matrix as sum of all entries in all matrices
    matrices = np.array(matrices)
    total_confusion_matrix = np.sum(matrices, axis=0)

    # Compute the metrics for the resulted total_confusion_matrix by calling appropriate methods
    # We have the accuracy and a mean of all the precisions, recalls and f1
    mean_depth = depth_sum / k
    precision = get_precision(total_confusion_matrix)
    recall = get_recall(total_confusion_matrix)
    f1 = get_f1(total_confusion_matrix)
    accuracy = get_accuracy(total_confusion_matrix)

    return precision, recall, f1, accuracy, mean_depth


# Given a trained decision tree, calculates the test_labels and predicted_labels and generates confusion matrix for them
def generate_conf_matrix_from_decision_tree(test_dataset, trained_tree):
    (test_labels, predicted_labels) = evaluate(test_dataset, trained_tree)
    return generate_confusion_matrix(test_labels, predicted_labels)


# Returns (precision, recall, f1, accuracy) as the mean of precisions, recalls, f1 of the 4 classes
# This method uses a validation dataset and is used for Step 4 only
# It is similar to get_cross_validation_metrics(k, dataset) but now we have also an inner cross valdiation:
# One fold for testing, one fold for validation and (k - 2) folds for training
def get_cross_validation_metrics_with_pruning(k, dataset):
    depth_sum = 0
    k = min(k, dataset.shape[0])
    split_indices = k_fold_split(k, dataset)
    matrices = []

    for i in range(k):
        test_indices = split_indices[i]
        test_dataset = dataset[test_indices, :]

        remaining_indices = split_indices[:i] + split_indices[i + 1:]

        # Inner cross validation
        for j in range(k - 1):
            # Get validation indices for fold j in remaining_indices
            validation_indices = remaining_indices[j]
            validation_dataset = dataset[validation_indices, :]

            # Get training indices for the rest of the folds (besides j) in remaining_indices
            train_indices = np.hstack(remaining_indices[:j] + remaining_indices[j + 1:])
            train_dataset = dataset[train_indices, :]

            # Train the tree on the train dataset
            (trained_tree, _) = decision_tree_learning(train_dataset, 0)
            depth_sum += get_depth(trained_tree)

            # Prune the tree on the validation dataset to get max possible accuracy
            pruned_tree = prune(trained_tree, None, None, trained_tree, validation_dataset)

            # Calculate confusion matrix of the pruned_tree on test_dataset and append to 'matrices'
            conf_matrix = generate_conf_matrix_from_decision_tree(test_dataset, pruned_tree)
            matrices.append(conf_matrix)

    # Calculate total_confusion_matrix as sum of all entries in all matrices
    matrices = np.array(matrices)
    total_confusion_matrix = np.sum(matrices, axis=0)

    # Compute the metrics for the resulted total_confusion_matrix by calling appropriate methods
    # We have the accuracy and and a mean of all the precisions, recalls and f1
    mean_depth = depth_sum / (k*(k-1))
    precision = get_precision(total_confusion_matrix)
    recall = get_recall(total_confusion_matrix)
    f1 = get_f1(total_confusion_matrix)
    accuracy = get_accuracy(total_confusion_matrix)

    return precision, recall, f1, accuracy, mean_depth


# Function which computes the metrics given the filepath and the number of folds k
def get_metrics_for_file(filepath, k, pruning):
    dataset = read_dataset(filepath)

    if pruning == Pruning.WITH_PRUNING:
        precision, recall, f1, accuracy, mean_depth = get_cross_validation_metrics_with_pruning(k, dataset)
    else:
        precision, recall, f1, accuracy, mean_depth = get_cross_validation_metrics(k, dataset)

    print(pruning)
    print("For: " + filepath)
    print("accuracy = " + str(accuracy))
    print("precision = " + str(precision))
    print("recall = " + str(recall))
    print("f1 = " + str(f1))
    print("mean depth = " + str(mean_depth))

    return precision, recall, f1, accuracy


# Implemente an evaluation function that takes a trained tree and a test dataset: evaluate(test_db, trained_tree)
# and that returns the accuracy of the tree

# Constructs the predicted labels by traversing the tree for each instance in the test dataset
# Returns the test_labels and predicted_labels
def evaluate(test_db, trained_tree):
    predicted_labels = np.array(list(map(lambda instance: trained_tree.get_trained_label(instance), test_db)))
    test_labels = test_db[:, -1]

    return test_labels, predicted_labels


# Returns precision array for a confusion matrix by the given formula : tp / (tp + fp)
def get_precision(confusion_matrix):
    p = np.zeros(confusion_matrix.shape[1], )

    # By considering each of the 4 classes as the positive one by one
    # We compute the precision and return all 4 of them as an array p
    for c in range(confusion_matrix.shape[0]):
        # tp is always the element on the diagonal for that class, represented by [c, c]
        tp = confusion_matrix[c, c]

        # tp + fp is the sum of entries in the current column
        column_sum = np.sum(confusion_matrix[:, c])
        if column_sum > 0:
            p[c] = tp / column_sum

    return p


# Returns recall array for a confusion matrix by the given formula : tp / (tp + fn)
def get_recall(confusion_matrix):
    r = np.zeros(confusion_matrix.shape[1], )

    # By considering each of the 4 classes as the positive one by one
    # We compute the recall and return all 4 of them as an array p
    for c in range(confusion_matrix.shape[0]):
        # tp is always the element on the diagonal for that class, represented by [c, c]
        tp = confusion_matrix[c, c]

        # tp + fn is the sum of entries in the current row
        row_sum = np.sum(confusion_matrix[c, :])
        if row_sum > 0:
            r[c] = tp / row_sum

    return r


# Returns accuracy for a confusion matrix by the given formula : (tp + tn) / (tp + tn + fp + fn)
# Which represents the sum of elements on the diagonal over the sum of all entries in the confusion matrix
def get_accuracy(confusion_matrix):
    matrix_sum = np.sum(confusion_matrix)
    diag_sum = np.sum(np.diag(confusion_matrix))

    if matrix_sum > 0:
        return diag_sum / matrix_sum
    else:
        return 0


# Returns f1 for a confusion matrix by the given formula : 2 * (p * r) / (p + r) where p = precision, r = recall
def get_f1(confusion_matrix):
    recall = get_recall(confusion_matrix)
    precision = get_precision(confusion_matrix)

    return np.array(list(map(lambda p, r: ((2 * p * r) / (p + r)) if (p + r > 0) else 0, precision, recall)))


get_metrics_for_file(fp_clean_dataset, 10, Pruning.WITHOUT_PRUNING)
get_metrics_for_file(fp_noisy_dataset, 10, Pruning.WITHOUT_PRUNING)
get_metrics_for_file(fp_clean_dataset, 10, Pruning.WITH_PRUNING)
get_metrics_for_file(fp_noisy_dataset, 10, Pruning.WITH_PRUNING)

