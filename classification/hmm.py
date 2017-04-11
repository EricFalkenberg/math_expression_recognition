import argparse
import numpy as np
from hmmlearn import hmm
from config import hmm_model_meta, arg_dataset, arg_classes
from features import extract_features

NUM_STATES = 6
NUM_GAUSSIANS = 5


classNames = ['!', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5',
              '6', '7', '8', '9', '=', 'A', 'B', 'C', 'COMMA', 'E', 'F', 'G',
              'H', 'I', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Y', '[',
              '\\Delta', '\\alpha', '\\beta', '\\cos', '\\div', '\\exists',
              '\\forall', '\\gamma', '\\geq', '\\gt', '\\in', '\\infty', '\\int',
              '\\lambda', '\\ldots', '\\leq', '\\lim', '\\log', '\\lt', '\\mu',
              '\\neq', '\\phi', '\\pi', '\\pm', '\\prime', '\\rightarrow',
              '\\sigma', '\\sin', '\\sqrt', '\\sum', '\\tan', '\\theta', '\\times',
              '\\{', '\\}', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i',
              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z', '|']


def trainModel(hmm, sample):
    hmm.predict(sample)
    return hmm


if __name__ == '__main__':
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description=hmm_model_meta['program_description'])
    parser.add_argument('dataset', **arg_dataset)
    parser.add_argument('classes', **arg_classes)
    args = parser.parse_args()

    hmmMap = {}
    for c in classNames:
        model = hmm.GMMHMM(n_components=NUM_STATES, n_mix=NUM_GAUSSIANS, covariance_type="diag")
        model.startprob_ = np.array([1, 0, 0, 0, 0, 0])
        model.transmat_  = np.array([[ 1.0/NUM_STATES ] * NUM_STATES ] * NUM_STATES)
        model.means_     = np.array([0] * NUM_STATES)
        print(model.transmat_)
        print(len(model.transmat_))
        #gt, features = extract_features(args.dataset[0])
        hmmMap[c] = model#trainModel(model, features)

    print(hmmMap.keys())



