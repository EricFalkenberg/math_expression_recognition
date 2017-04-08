## KD-Tree Specific Configuration
kdtree_meta = {
    'program_description': """
                           Classify isolated handwritten math symbols with a k-d tree implementation of
                           the 1-NN algorithm.
                           """
}
kdtree_model = {
    'leaf_size' : 40
}

## General Configuration
dataset_config = {
    'metavar' : 'f',
    'type'  : str,
    'nargs' : 1,
    'help'  : 'The filename for the csv file that contains the dataset.'
}
classes_config = {
    'metavar' : 'c',
    'type' : str,
    'nargs' : '*',
    'help' : 'The class(es) that should be processed by the classifier'
}
