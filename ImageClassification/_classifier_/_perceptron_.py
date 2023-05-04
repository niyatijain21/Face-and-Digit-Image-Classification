import random
import numpy as np
import math

#Check
""" 
digit width = 28
digit height = 28
face width = 60
face height = 70
"""
class Perceptron:
    def __init__(self, legal_labels):
        self.legalLabels = legal_labels
        self.weights = {}
        self.bias = {}
        self.iterations = 0
        digit = True if len(legal_labels) == 10 else False

        if digit:
            for label in legal_labels:
                self.weights[label] = np.array([random.uniform(0, 1) for i in range(28 * 28)])
                self.bias[label] = random.uniform(0, 1)
        else:
            for label in legal_labels:
                self.weights[label] = np.array([random.uniform(0, 1) for i in range(70 * 60)])
                self.bias[label] = random.uniform(0, 1)
                        
    @staticmethod
    def name():
        return "Perceptron"

    def calculate_activation(self, data, bias, weights):
        fx = bias + np.dot(data, weights)
        return 1 / (1 + math.exp(-0.1 * fx))

    def train(self, training_data, validation_data):
        classes = {}
        inverse_classes = {} 
        for label in self.legalLabels:
            classes[label] = []
            inverse_classes[label] = []

        for x, y in training_data.get_labeled_images():
            x = np.array(x.flat_data())
            classes[y].append(x)
            for label in self.legalLabels:
                if label != y:
                    inverse_classes[label].append(x)

        alpha = 0.05
        for i in range(0,10):
            for label in self.legalLabels:
                for image in classes[label]:
                    error = 1 - self.calculate_activation(image, self.bias[label], self.weights[label])
                    self.bias[label] += alpha * error
                    self.weights[label] = self.weights[label] + alpha * error * image

                for image in inverse_classes[label]:
                    error = 0 - self.calculate_activation(image, self.bias[label], self.weights[label])
                    self.bias[label] += alpha * error
                    self.weights[label] += alpha * error * image

    def classify(self, data):
        predictions = []
        for image, label in data.get_labeled_images():
            flat_image = image.flat_data()
            activationvalues = []
            for y in self.weights.keys():
                activation = self.calculate_activation(flat_image, self.bias[y], self.weights[y])
                activationvalues.append((y, activation))
            predictions.append(max(activationvalues, key=lambda x: x[1])[0])
            compiled_tuple = tuple(zip(predictions, data.get_labels()))
            correct = sum(map(lambda x: int(x[0] == x[1]), compiled_tuple))
        return correct/len(data)
            