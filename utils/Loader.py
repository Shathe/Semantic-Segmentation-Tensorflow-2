from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import glob
import cv2
from utils import LoaderQueue
import time
import threading

class Loader:
    def __init__(self, dataFolderPath, width=224, height=224, n_classes=21, median_frequency=0.):
        '''
        Initializes the Loader

        :param dataFolderPath: Path to the dataset
        :param width: with to load the images
        :param height: height to load the images
        :param n_classes: number of classes of the dataset
        :param median_frequency: factor to power the median frequency balancing (0 to none effect, 1 to full efect)
        '''

        self.dataFolderPath = dataFolderPath
        self.height = height
        self.width = width
        self.dim = 3
        self.freq = np.zeros(n_classes)  # vector for calculating the class frequency
        self.index_train = 0  # indexes for iterating while training
        self.index_test = 0  # indexes for iterating while testing
        self.median_frequency_soft = median_frequency  # softener value for the median frequency balancing (if median_frequency==0, nothing is applied, if median_frequency==1, the common formula is applied)
        self.lock = threading.Lock()
        print('Reading files...')

        # Load filepaths
        files = glob.glob(os.path.join(dataFolderPath, '*', '*', '*'))

        print('Structuring test and train files...')
        self.test_list = [file for file in files if 'test' in file]
        self.train_list = [file for file in files if 'train' in file]

        '''
        The structure has to be dataset/train/images/image.png
        The structure has to be dataset/train/labels/label.png
        Separate image and label lists
        Sort them to align labels and images
        '''

        self.image_train_list = [file for file in self.train_list if 'images' in file]
        self.image_test_list = [file for file in self.test_list if 'images' in file]
        self.label_train_list = [file for file in self.train_list if 'labels' in file]
        self.label_test_list = [file for file in self.test_list if 'labels' in file]

        self.label_test_list.sort()
        self.image_test_list.sort()
        self.label_train_list.sort()
        self.image_train_list.sort()

        # Shuffle train
        self.suffle_segmentation()

        print('Loaded ' + str(len(self.image_train_list)) + ' training samples')
        print('Loaded ' + str(len(self.image_test_list)) + ' testing samples')
        self.n_classes = n_classes

        if self.median_frequency_soft != 0:
            self.median_freq = self.median_frequency_balancing_sof(soft=self.median_frequency_soft)

        # Creates test and train queues
        self.test_queue = LoaderQueue.LoaderQueue(100, self, train=False, workers=1)
        self.train_queue = LoaderQueue.LoaderQueue(100, self, train=True, workers=8)


        print('Dataset contains ' + str(self.n_classes) + ' classes')

    def suffle_segmentation(self):
        '''
        Shuffles the training files
        :return:
        '''
        s = np.arange(len(self.image_train_list))
        np.random.shuffle(s)
        self.image_train_list = np.array(self.image_train_list)[s]
        self.label_train_list = np.array(self.label_train_list)[s]

    def _from_binarymask_to_weighted_mask(self, labels, masks):
        '''
        This function updates the masks images using the median frequency balacing

        :param labels: label images
        :param masks: an array of N binary masks 0/1 of size [N, H, W ] where the 0 are pixeles to ignore from the labels [N, H, W ]
        and 1's means pixels to take into account.

        :return: The updated masks, weighted with the median frequency balacing
        '''

        weights = self.median_freq

        for i in range(masks.shape[0]):
            # for every mask of the batch
            label_image = labels[i, :, :]
            mask_image = masks[i, :, :]
            dim_1 = mask_image.shape[0]
            dim_2 = mask_image.shape[1]
            label_image = np.reshape(label_image, (dim_2 * dim_1))
            mask_image = np.reshape(mask_image, (dim_2 * dim_1))

            for label_i in range(self.n_classes):
                # multiply the mask so far, with the median frequency wieght of that label
                # print('label_i')
                # print(weights[label_i])
                # print(mask_image[label_image == label_i] )
                mask_image[label_image == label_i] = mask_image[label_image == label_i] * weights[label_i]
                # print(mask_image[label_image == label_i] )

            # unique, counts = np.unique(mask_image, return_counts=True)

            mask_image = np.reshape(mask_image, (dim_1, dim_2))
            masks[i, :, :] = mask_image

        return masks

    def get_data_list_and_index(self, train=True):
        '''

        :param train: whether to get training samples of testing samples
        :param size: size of the batch

        :return: image and label lists from where to load the images
        '''

        self.lock.acquire()

        if train:
            image_list = self.image_train_list
            label_list = self.label_train_list

            # Get [size] indexes
            index = self.index_train
            self.index_train = (index + 1) % len(image_list)
        else:
            image_list = self.image_test_list
            label_list = self.label_test_list

            # Get [size] random numbers
            index = self.index_test
            self.index_test = (index + 1) % len(image_list)

        self.lock.release()

        return image_list, label_list, index


    def load_image(self, file_path):
        '''

        :param file_path: path to the image
        :return: return the loaded image
        '''

        if self.dim == 1:
            img = cv2.imread(file_path, 0)
        else:
            # img = cv2.imread(random_images[index])
            img = tf.keras.preprocessing.image.load_img(file_path)
            img = tf.keras.preprocessing.image.img_to_array(img).astype(np.uint8)

        return img

    def load_label(self, file_path):
        '''
        :param file_path: path to the label image
        :return: return the loaded label image
        '''
        return cv2.imread(file_path, 0)




    def get_sample(self, train=True):
        '''
        Get a sample of the segmentation dataset

        :param train: whether to get training samples of testing samples
        :param labels_resize_factor: (downsampling) factor to resize the label images

        :return: sample of segmentation images: X, labels: Y and, masks: mask
        '''


        # init numpy arrays
        image_list, label_list, index = self.get_data_list_and_index(train)

        # for every image, get the image, label and mask.
        # the augmentation has to be done separately due to augmentation
        img = self.load_image(image_list[index])
        label = self.load_label(label_list[index])
        mask_image = np.ones([self.height, self.width], dtype=np.float32)

        # Reshape images if its needed
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if label.shape[1] != self.width or label.shape[0] != self.height:
            label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)


        # modify the mask and the labels. Mask
        mask_ignore = label >= self.n_classes
        mask_image[mask_ignore] = 0  # The ignore pixels will have a value o 0 in the mask
        label[mask_ignore] = 0  # The ignore label will be n_classes

        if self.dim == 1:
            img = np.reshape(img, (img.shape[0], img.shape[1], self.dim))



        y = np.expand_dims(label, axis=0)
        mask = np.expand_dims(mask_image, axis=0)

        # Apply weights to the mask
        if self.median_frequency_soft > 0:
            mask = self._from_binarymask_to_weighted_mask(y, mask)

        # the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
        a, b, c = y.shape
        y = y.reshape((a * b * c))

        # Convert to categorical. Add one class for ignored pixels
        y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
        y = y.reshape((a, b, c, self.n_classes)).astype(np.uint8)



        return img, y[0, ...], mask[0, ...]

    def get_queue(self, train=True):
        '''

        :param train: wheter to get the training or testing queue
        :return: LoaderQueue
        '''
        if train:
            return self.train_queue
        else:
            return self.test_queue


    def get_batch(self, size=32, train=True):
        '''
        Get a batch of the segmentation dataset

        :param size: size of the batch
        :param train: whether to get training samples of testing samples

        :return: batch of segmentation images: X, labels: Y and, masks: mask
        '''

        queue = self.get_queue(train)

        # init numpy arrays
        x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
        y = np.zeros([size, self.height, self.width, self.n_classes], dtype=np.uint8)
        mask = np.ones([size, self.height, self.width], dtype=np.float32)

        for index in range(size):
            img, label, mask_image = queue.__next__()

            x[index, :, :, :] = img.astype(np.float32)
            y[index, :, :] = label
            mask[index, :, :] = mask_image

        return x, y, mask

    def median_frequency_balancing_sof(self, soft=1):
        '''

        :param soft: softening factor to power the median frequency
        :return: A vector of size [self.n_classes] where each class has a weight computed through
        the median frequency balancing
        '''
        for image_label_train in self.label_train_list:
            image = cv2.imread(image_label_train, 0)
            for label in range(self.n_classes):
                self.freq[label] = self.freq[label] + sum(sum(image == label))

        # Common code
        zeros = self.freq == 0
        if sum(zeros) > 0:
            print('There are some classes which are not contained in the training samples')

        results = np.median(self.freq) / self.freq
        results[zeros] = 0  # for not inf values.
        results = np.power(results, soft)
        print(results)
        return results

    '''
    # Called when iteration is initialized
    def __iter__(self):
        self.index_train = 0  # indexes for iterating while training
        self.index_test = 0  # indexes for iterating while testing
        return self
        
    def __getitem__(self, item):
        pass

    def __next__(self):
        # obtener el siguiente usando get item (dado self.index_train) y haciendo ++

        pass
    '''

if __name__ == "__main__":

    loader = Loader('./Datasets/camvid', n_classes=11, width=480, height=360, median_frequency=0.12)
    # print(loader.median_frequency_exp())
    x, y, mask = loader.get_batch(size=2)

    for i in range(2):
        cv2.imshow('x', ((x[i, :, :, :] + 1) * 127.5).astype(np.uint8))
        cv2.imshow('y', (np.argmax(y, 3)[i, :, :] * 25).astype(np.uint8))
        print(mask.shape)
        cv2.imshow('mask', (mask[i, :, :] * 255).astype(np.uint8))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    x, y, mask = loader.get_batch(size=3, train=False)
