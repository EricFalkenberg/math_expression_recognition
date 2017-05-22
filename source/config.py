baseline_segmenter_meta = {
    'program_description' : """
                            Segmentation algorithm that treats every stroke as a symbol and then
                            attempts to classify based on the random forest classifier in
                            the classification directory.
                            """
}

parser_meta = {
    'program_description' : """
                            Parsing algorithm that uses an MST based aproach to perform structural
                            analysis on groupings of math symbols.
                            """
}

segmenter_meta = {
    'program_description' : """
                            Segmentation algorithm that uses Adaboost trained on a combination of 
                            geometric, multi-scale shape context, and classifier features.
                            """
}

file_handler_config = {
    'training_data_full' : [
            "/home/eric/Desktop/pattern_rec/TrainINKML/expressmatch",
            "/home/eric/Desktop/pattern_rec/TrainINKML/extension",
            "/home/eric/Desktop/pattern_rec/TrainINKML/HAMEX",
            "/home/eric/Desktop/pattern_rec/TrainINKML/MathBrush",
            "/home/eric/Desktop/pattern_rec/TrainINKML/MfrDB",
            "/home/eric/Desktop/pattern_rec/TrainINKML/KAIST"

    ],
    'training_data_medium' : [
            "/home/eric/Desktop/pattern_rec/TrainINKML/HAMEX"
    ],
    'training_data_small' : [
            "/home/eric/Desktop/pattern_rec/TrainINKML/expressmatch"
    ],
    'training_data_tiny' : [
            "/home/eric/Desktop/pattern_rec/TrainINKML/extension"
    ]
}

arg_data_type = {
    'metavar' : 'type',
    'type'  : str,
    'nargs' : 1,
    'help'  : 'Determine what size of data to train on',
    'choices' : ['tiny', 'small', 'medium', 'full']
}
arg_command = {
    'metavar' : 'cmd',
    'type'  : str,
    'nargs' : 1,
    'help'  : 'Whether to test or train the model',
    'choices' : ['train', 'test', 'test_segmenter']
}
