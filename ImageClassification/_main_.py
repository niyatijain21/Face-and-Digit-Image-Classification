import argparse
import datasets
import sys
import random

from _classifier_ import *
from time import monotonic
from images import Datum

def face_datasets():
    return [
        datasets.load_dataset("facedata/facedatatrain", "facedata/facedatatrainlabels", 70),
        datasets.load_dataset("facedata/facedatavalidation", "facedata/facedatavalidationlabels", 70),
        datasets.load_dataset("facedata/facedatatest", "facedata/facedatatestlabels", 70),
        "faces"
    ]


def digit_datasets():
    return [
        datasets.load_dataset("digitdata/trainingimages", "digitdata/traininglabels", 28),
        datasets.load_dataset("digitdata/validationimages", "digitdata/validationlabels", 28),
        datasets.load_dataset("digitdata/testimages", "digitdata/testlabels", 28),
        "digits"
    ]


def feature_extractor(image):
    image = image.data()
    for y in range(len(image)):
        for x in range(len(image[y])):
            if image[y][x] > 0:
                image[y][x] = 1
    return Datum(image)


def run_test(datasets, algorithm_name, algorithm):
    training, validation, test, dataname = datasets
    training_percentage = map(lambda x: x/10, range(1, 11))

    training.extract(feature_extractor)
    test.extract(feature_extractor)

    final = []
    for p in training_percentage:
        training_size = int(p * len(training))
        for iter in range(0,5):
            training_set = training.subset(training_size, len(training)) 
            start_time = monotonic()
            if algorithm==3: 
                clf = algorithm_name.train(training_set)
            else:
                algorithm_name.train(training_set, validation)
            end_time = monotonic()

            if algorithm==3:
                accuracy = algorithm_name.classify(test, clf)
            else:
                accuracy = algorithm_name.classify(test)
            final.append((dataname, training_size, iter+1, end_time - start_time, accuracy)) 
    return final


def main():
    parser = argparse.ArgumentParser(description="CS520 Image Classification Project by Niyati and Agrani")
    parser.add_argument("-a", "--algorithm", type=int, choices=[1, 2, 3],
                        help="1. Perceptron, 2. Naive Bayes Classifier, 3. SVM", default=None)
    parser.add_argument("-d", "--data", type=str, choices=["face", "digit"],
                        help="whether to evaluate faces or digits", default=None)
    args = vars(parser.parse_args())


    algorithm = args["algorithm"]
    datasource = args["data"]
    algorithms = [Perceptron, NaiveBayes, SVM]
    final = {}

    if not (algorithm or datasource):
        print()
        print("Must specify both -a and -d.")
        parser.print_help(sys.stdout)
        return

    elif not (algorithm and datasource):
        print("Must specify both -a and -d.\nExiting.")

    else:
        data = {"face": face_datasets, "digit": digit_datasets}[datasource]()
        labels = {"face": [0, 1], "digit": list(range(0, 10))}[datasource]
        algorithm_name = algorithms[algorithm-1](labels)
        final[algorithm_name.name()] = run_test(data, algorithm_name, algorithm)

    for algorithm in final.keys():

        if len(final[algorithm]) == 0:
            continue

        for entry in final[algorithm]:
            print("Algorithm:     ", algorithm)
            print("Data type:     ", entry[0])
            print("Training size: ", entry[1])
            print("Iterations:    ", entry[2])
            print("Training time: ", entry[3])
            print("Accuracy:      ", entry[4], "\n")


if __name__ == "__main__":
    main()

