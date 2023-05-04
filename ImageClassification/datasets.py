
import images
import random

class Dataset:
    def __init__(self, imgs, labels):
        self._images = imgs
        self._labels = labels
        self._pairs = tuple(zip(self._images, self._labels))

        self._associations = dict(self.get_labeled_images())

    def __len__(self):
        return len(self._images)

    def __iter__(self):
        return self.get_labeled_images().__iter__()

    def subset(self, n, high):
        arr = random.sample(range(0, high), n)
        new_images = []
        new_labels = []
        for r in arr:
            new_images.append(self._images[r])
            new_labels.append(self._labels[r])
        return Dataset(new_images, new_labels)

    def get_labeled_images(self):
        return self._pairs

    def extract(self, extractor):
        for i in range(len(self._images)):
            self._images[i] = extractor(self._images[i])

    def get_images(self):
        return self._images.copy()

    def get_labels(self):
        return self._labels.copy()

    def get_image(self, index):
        return self._images[index]

    def get_label(self, index):
        return self._labels[index]

    def get_image_label(self, image):
        return self._associations[image]


def load_dataset(imagefile, labelfile, imageheight):
    image_data = images.load_images(imagefile, imageheight)
    label_data = []

    with open(labelfile) as file:
        for line in file:
            label_data.append(int(line))

    return Dataset(image_data, label_data)
