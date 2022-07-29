import matplotlib.pyplot as plt

from Datasets import fp_dummy_example
from DecisionTree import *

space_between_nodes_X = 30
space_between_nodes_Y = 0
long_version = False
font_size = 10

def visualise(tree, depth):
    global space_between_nodes_Y
    space_between_nodes_Y = depth * 2

    margin_top = 30

    width = 200
    height = 200

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)

    # Set title for the figure
    fig.suptitle('Decision Tree Visualisation, depth = ' + str(depth), fontsize=font_size, fontweight='bold')

    # Set the width and height
    ax.axis([0, width, 0, height])

    # Call method to populate the plot
    create_visualisation(tree, ax, width / 2, height - margin_top, 0, depth)

    # We don't want coordinates to be displayed on X and Y axis
    plt.axis('off')

    plt.show()


# Creates the visualisation of the tree, i.e. the nodes and leaves with arrows pointing between them
def create_visualisation(tree, ax, currX, currY, currDepth, totalDepth):
    # TODO: comment this function
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    nodes = totalDepth - currDepth

    if isinstance(tree, Leaf):
        label = int(tree.label)
        if long_version:
            str_val = 'leaf: ' + str(label)
        else:
            str_val = str(label)
        ax.text(currX, currY, str_val, ha="center", va="center",  color='red', size=10,
                bbox=bbox_props)
    elif isinstance(tree, Node):
        att = str(tree.attribute)
        val = str(tree.value)

        if long_version:
            str_val = '[x' + att + ' < ' + val + ']'
        else:
            str_val = str(att) + " " + val
        ax.text(currX, currY, str_val, ha="center", va="center", color='purple', size=10,
                bbox=bbox_props)
        
        # if it is the first node
        if currDepth == 0:
            currDepth = 1.5        

        nextXLeft = currX - space_between_nodes_X/((currDepth/1.5)**2)*2
        nextYLeft = currY - space_between_nodes_Y

        nextXRight = currX + space_between_nodes_X/((currDepth/1.5)**2)*2
        nextYRight = currY - space_between_nodes_Y

        # add arrow to the left
        ax.annotate('', xy=(nextXLeft, nextYLeft), xytext=(currX, currY - 1),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        # add arrow to the right
        ax.annotate('', xy=(nextXRight, nextYRight), xytext=(currX, currY - 1),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        #recursive call
        create_visualisation(tree.left, ax, nextXLeft, nextYLeft, currDepth + 1, totalDepth)
        create_visualisation(tree.right, ax, nextXRight, nextYRight, currDepth + 1, totalDepth)

# dataset = read_dataset(fp_clean_dataset)
# (tree, depth) = decision_tree_learning(dataset, 0)
# visualise(tree, depth)
