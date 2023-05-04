import numpy as np

class NaiveBayes():
    def __init__(self, legal_labels):
        self.legalLabels = legal_labels
        self.occurences_of_labels = {}
        self.prob_of_labels = {}

    @staticmethod
    def name():
        return "Naive Bayes"

    def train(self, training_data, validation_data):
        classes = {}
        for x, y in training_data.get_labeled_images():
            if y not in classes:
                classes[y] = []
            classes[y].append(x)

        for label in classes:
            images = classes[label]              
            flat_images = tuple(map(lambda x: x.flat_data(), images)) 
            no_of_features = len(flat_images[0])    
            feature_sum = [0] * no_of_features             

            for image in flat_images:
                for i in range(no_of_features):
                    if image[i] > 0:
                        feature_sum[i] += 1

            alpha = 1
            for i in range(no_of_features):
                feature_sum[i] = (feature_sum[i] + alpha) / (len(images) + 2 * alpha)

            feature_sum = np.array(feature_sum)
            self.prob_of_labels[label] = {"p": feature_sum, "np": 1 - feature_sum}
            self.occurences_of_labels[label] = len(images) / len(training_data)

    def classify(self, data):
        predictions = []
        for image, label in data.get_labeled_images():
            fx = image.flat_data()
            probabilities = {}
            fx = np.array(fx)
            finversex = 1 - fx
            for label, probs in self.prob_of_labels.items():
                probabilities[label] = 1
                probabilities[label] *= np.log((probs["p"] * fx) + (probs["np"] * finversex)).sum()
            predictions.append(max(probabilities.items(), key=lambda x: x[1])[0])
            
            compiled_tuple = tuple(zip(predictions, data.get_labels()))
            correct = sum(map(lambda x: int(x[0] == x[1]), compiled_tuple))
        return correct/len(data)
