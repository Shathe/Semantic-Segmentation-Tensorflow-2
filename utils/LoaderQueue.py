from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
import glob
import cv2
import asyncio
import _thread
import time
import threading

class LoaderQueue:
    def __init__(self, size, loader, workers, train):
        '''
        Creates a queue for loading data

        :param size: size of the queue (max elements on the queue)
        :param loader: loader (Loader Class)
        :param workers: number of threads
        :param train: whether is training or testing
        '''
        self.size = size
        self.train = train
        self.loader = loader
        self.workers = workers
        self.x = []
        self.y = []
        self.mask = []
        self.lock = threading.Lock()

        for i in range(self.workers):
            _thread.start_new_thread(load, ("Thread" + str(i), self, ))



    def __next__(self):
        '''
        Returns an element of the queue. If the queue is empty, sleep some miliseconds and retry
        :return: an element of the queue (round robin)
        '''
        if len(self.x) > 0:
            return self.x.pop(0), self.y.pop(0), self.mask.pop(0)
        else:
            time.sleep(0.005)
            return self.__next__()



def load(name, queue):
    '''
    Infinite function, while the queue is not full, load  more elements

    :param name: name of the process
    :param queue: queue (LoaderQueue Class)
    :return:
    '''

    while True:
        if len(queue.x) < queue.size:
            x_i, y_i, mask_i = queue.loader.get_sample(train=queue.train) # Load sample

            # Save sample on queue (use lock for saving x, y, mask elements on the same index
            queue.lock.acquire()
            queue.x.append(x_i)
            queue.y.append(y_i)
            queue.mask.append(mask_i)
            queue.lock.release()

        else:
            time.sleep(0.01)




if __name__ == "__main__":
    pass
