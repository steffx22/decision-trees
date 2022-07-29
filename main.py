from Datasets import fp_clean_dataset
from DecisionTree import *

dataset = read_dataset(fp_clean_dataset)
decision_tree, depth = decision_tree_learning(dataset, 0)
