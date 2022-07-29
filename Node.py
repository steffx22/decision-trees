import numpy as np

class Tree:
    def get_trained_label(self, instance):
        pass


class Node (Tree):
    def __init__(self, value, left, right, attribute, depth):
        self.left = left            # Will contain the left dataset
        self.right = right          # Will contain the right dataset
        self.value = value          # Split point value
        self.attribute = attribute  # Attribute index ranging 0 .. 6
        self.depth = depth          # Depth from root to current node

    # Traverse tree to get the predicted label
    def get_trained_label(self, instance):
        attribute_value = instance[self.attribute]

        # We search on either left subtree (if currValue < nodeValue) or right subtree (if currValue >= nodeValue)
        if attribute_value < self.value:
            return self.left.get_trained_label(instance)
        return self.right.get_trained_label(instance)


class Leaf (Tree):
    def __init__(self, label, depth, no_instances):
        self.label = label          # Trained label
        self.depth = depth          # Depth from root to current leaf

        self.instances = np.zeros(4)
        self.instances[int(label) - 1] = no_instances

    # For a leaf we just return the trained label that is stored within the leaf
    def get_trained_label(self, instance):
        return self.label


