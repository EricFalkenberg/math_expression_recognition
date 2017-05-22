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

## Running the segmenter
```
usage: segmenter.py [-h] cmd type

Segmentation algorithm that uses Adaboost trained on a combination of
geometric, multi-scale shape context, and classifier features.

positional arguments:
  cmd         Whether to test or train the model
  type        Determine what size of data to train on

optional arguments:
  -h, --help  show this help message and exit
```

## Running the parser
```
usage: parser.py [-h] cmd type

Parsing algorithm that uses an MST based aproach to perform structural
analysis on groupings of math symbols.

positional arguments:
  cmd         Whether to test, test_segmenter, or train the model
  type        Determine what size of data to train on

optional arguments:
  -h, --help  show this help message and exit
```

It is worth noting that the `test_segmenter` option exists to allow you to test
parsing on segmentation output instead of ground truth. Though this is a requirement, 
the pretrained models do not support the functionality as the segmenter was trained in full
before the method allowing this functionality was added. If you would like to test this, you
must retrain the segmenter.

## Dataset
The classifiers currently expects the `trainingSymbols/` and `trainingJunk/` to reside under 
`dataset/`. To change this behavior, edit the location attribute of the dataset_meta dictionary in `classification/config.py`.

The segmenter and parser expect the dataset at whatever is specified in the base level config.py. Change this to point to your dataset.


## Libraries
This project uses the following libraries.
1. sklearn
2. hmmlearn
3. progressbar2
4. numpy
5. scipy
6. networkx
7. PIL
8. BeautifulSoup 

