# pattern_rec_project

Project I for Pattern Recognition @ RIT

## Running the classifiers

Both classifiers are found in the classification directory and can be run
using the following

```
usage: kd_tree.py [-h] cmd f [c [c ...]]

Classify isolated handwritten math symbols with a k-d tree implementation of
the 1-NN algorithm.

positional arguments:
  cmd         Whether to test or train the model
  f           The filename for the csv file that contains the dataset.
  c           The class(es) that should be processed by the classifier

optional arguments:
  -h, --help  show this help message and exit
```

```
usage: hmm.py [-h] cmd f [c [c ...]]

Classify isolated handwritten math symbols with an HMM.

positional arguments:
  cmd         Whether to test or train the model
  f           The filename for the csv file that contains the dataset.
  c           The class(es) that should be processed by the classifier

optional arguments:
  -h, --help  show this help message and exit
```

For instance, the `kd_tree` implementation can be trained
```
python kd_tree.py train path/to/training_set.csv
```
which will save a model at `classification/models/kd_tree.model` and then
tested
```
python kd_tree.py test path/to/testing_set.csv
```
which will load the model from this location.

Pre-trained models have been provided so you need only run the test command
to see our models in action.

## Dataset
The program currently expects the `trainingSymbols/` and `trainingJunk/` to reside under `dataset/`.
To change this behavior, edit the location attribute of the dataset_meta dictionary in `config.py`.
