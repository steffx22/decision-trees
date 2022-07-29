# Decision Trees

In order to run the code, the python packages numpy and matplotlib need to be installed

To add datasets:
 1. Place the dataset in the form of a text file in the folder wifi_db
 2. Add the path of the text file to Datasets.py and name your dataset

To display the metrics:
 1. At the bottom of the Metrics.py file, add the following lines, replacing your_dataset with the desired dataset:\
   a. get_metrics_for_file(your_dataset, 10, Pruning.WITHOUT_PRUNING) for metrics before pruning\
   b. get_metrics_for_file(your_dataset, 10, Pruning.WITH_PRUNING) for metrics after pruning

To visualise the decisionTree:
 1. At the bottom of the Visualisation.py, add the following lines, replacing your_dataset with the desired dataset:\
 dataset = read_dataset(your_dataset), to parse the dataset\
 (tree, depth) = decision_tree_learning(dataset, 0), to make the tree\
 visualise(tree, depth), to draw the tree
