import numpy as np

## KD-Tree Specific Configuration
kdtree_meta = {
    'program_description': """
                           Classify isolated handwritten math symbols with a k-d tree implementation of
                           the 1-NN algorithm.
                           """
}
kdtree_model = {
    'leaf_size' : 40,
    'metric' : 'minkowski'
}

## HMM Specific Configuration
random_forest_meta = {
    'program_description': """
                           Classify isolated handwritten math symbols with an HMM.
                           """,
    'class_names' : ['!', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5',
                      '6', '7', '8', '9', '=', 'A', 'B', 'C', 'COMMA', 'E', 'F', 'G',
                      'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[',
                      '\\Delta', '\\alpha', '\\beta', '\\cos', '\\div', '\\exists',
                      '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int',
                      '\\lambda', '\\ldots', '\\leq', '\\lim', '\\log', '\\lt', '\\mu',
                      '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow',
                      '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times',
                      '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i',
                      'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                      'v', 'w', 'x', 'y', 'z', '|', 'junk'],
    'num_features' : [4 for _ in range(55)]
}
random_forest_model = {
    'n_components': 6,
    'n_mix' : 5
}

## General Configuration
arg_command = {
    'metavar' : 'cmd',
    'type'  : str,
    'nargs' : 1,
    'help'  : 'Whether to test or train the model',
    'choices' : ['train', 'test']
}
arg_dataset = {
    'metavar' : 'f',
    'type'  : str,
    'nargs' : 1,
    'help'  : 'The filename for the csv file that contains the dataset.'
}
arg_classes = {
    'metavar' : 'c',
    'type' : str,
    'nargs' : '*',
    'help' : 'The class(es) that should be processed by the classifier'
}

## Dataset Configuration
dataset_meta = {
    'location' : '../dataset',
    'exclude'  : ['iso_GT.txt', 'junk_GT.txt'],
    'xml_name_tag' : '{http://www.w3.org/2003/InkML}annotation',
    'xml_trace_tag' : '{http://www.w3.org/2003/InkML}trace'
}
