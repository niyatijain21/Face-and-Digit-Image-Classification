import numpy as np
from sklearn import svm


class SVM:
    def __init__(self, legal_labels):
        self.legalLabels = legal_labels
        self.final_data = []
        self.final_labels = []
        self.count = 0
        
    @staticmethod
    def name():
        return "Support Vector Machines"

    def train(self, training_data):
        count = 0
        final_data = []
        final_labels = []

        classes = {}
        for x, y in training_data.get_labeled_images():
            if y not in classes:
                classes[y] = []
            classes[y].append(x)  

        for y in classes:
            images = classes[y]               
            flat_images = tuple(map(lambda x: x.flat_data(), images))
            for x in flat_images:
                count += 1
                final_data.append(x)
                final_labels.append(y)

        self.final_data = final_data
        self.final_labels = final_labels
        self.count = count
        final_data = np.array(final_data, dtype=np.int32)
        final_labels = np.array(final_labels, dtype=np.int32)

        clf = svm.SVC(kernel='linear')
        clf.fit(final_data,final_labels)
        return clf

    def classify(self, data, clf):
        test_array = []
        predictions = []
        count = 0
        flat_images = tuple(map(lambda x: x.flat_data(), data.get_images()))  # flat images
        for img in flat_images:
            count += 1
            test_array.append(img)
        test_array = np.array(test_array, dtype=np.int32)
        predictions = clf.predict(test_array)
        compiled_tuple = tuple(zip(predictions, data.get_labels()))
        correct = sum(map(lambda x: int(x[0] == x[1]), compiled_tuple))

        return correct/len(data)