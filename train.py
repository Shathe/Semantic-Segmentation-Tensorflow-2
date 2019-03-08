import numpy as np
import tensorflow as tf
import nets.MiniNetv2 as MiniNetv2
import os
import utils.Loader as Loader
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, erase_ignore_pixels, init_model, get_metrics
import cv2
tf.random.set_seed(7)
np.random.seed(7)

# @tf.function
def train_step(model, x, y, mask, loss_function, optimizer):
    with tf.GradientTape() as tape:


        [x, y, mask] = convert_to_tensors([x, y, mask]) # convert numpy to tensor
        tf.keras.backend.set_learning_phase(True) # set training phase
        y_ = model(x)  # get output of the model.

        # Apply mask to ignore labels
        mask = tf.expand_dims(mask, axis=-1)
        y *= mask

        loss = loss_function(y, y_) # apply loss
        #print(loss.numpy())

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))# Apply gradientes





# Trains the model for certains epochs on a dataset
def train(loader, optimizer, loss_function, model, epochs=5, batch_size=2, augmenter=False, lr=None, init_lr=2e-4,
          evaluation=True, name_best_model = 'weights/best', preprocess_mode=None, labels_resize_factor=1):
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / batch_size) + 1
    best_miou = 0

    for epoch in range(epochs):  # for each epoch
        lr_decay(lr, init_lr, 1e-9, epoch, epochs - 1)  # compute the new lr
        print('epoch: ' + str(epoch+1) + '. Learning rate: ' + str(lr.numpy()))
        for step in range(steps_per_epoch):  # for every batch
            # get batch
            x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter, labels_resize_factor=labels_resize_factor)
            x = preprocess(x, mode=preprocess_mode)

            train_step(model, x, y, mask, loss_function, optimizer)

        if evaluation:
            # get metrics
            tf.keras.backend.set_learning_phase(True)
            train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode, labels_resize_factor=labels_resize_factor)
            tf.keras.backend.set_learning_phase(False)
            test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,
                                              scales=[1], preprocess_mode=preprocess_mode, labels_resize_factor=labels_resize_factor)

            print('Train accuracy: ' + str(train_acc.numpy()))
            print('Train miou: ' + str(train_miou.numpy()))
            print('Test accuracy: ' + str(test_acc.numpy()))
            print('Test miou: ' + str(test_miou.numpy()))
            print('')

            # save model if bet
            if test_miou > best_miou:
                best_miou = test_miou
                model.save_weights(name_best_model)
        else:
            model.save_weights(name_best_model)

        loader.suffle_segmentation()  # sheffle trainign set


if __name__ == "__main__":
    # PARAMETERS

    # GPU to use
    n_gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)

    n_classes = 11
    batch_size = 4
    epochs = 1000
    width = 960
    height = 720
    labels_resize_factor = 2 # factor to divide the label size
    channels = 3
    lr = 8e-4
    name_best_model = 'weights/camvid/best'
    dataset_path = 'Datasets/camvid'
    preprocess_mode = 'imagenet'  #possible values 'imagenet', 'normalize',None

    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation',
                           width=width, height=height, channels=channels)

    # build model
    model = MiniNetv2.MiniNetv2(num_classes=n_classes)

    # optimizer
    learning_rate = tf.Variable(lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_function = tf.losses.CategoricalCrossentropy()

    # restore if model saved and show number of params
    restore_state(model, name_best_model)

    train(loader=loader, optimizer=optimizer, loss_function=loss_function, model=model, epochs=epochs, batch_size=batch_size,
          augmenter='segmentation', lr=learning_rate, init_lr=lr,  name_best_model=name_best_model, evaluation=True, preprocess_mode=preprocess_mode,
          labels_resize_factor=labels_resize_factor)

    get_params(model)

    print('Testing model')
    test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1],
                                      write_images=True, preprocess_mode=preprocess_mode, time_exect=False, labels_resize_factor=labels_resize_factor)
    print('Test accuracy: ' + str(test_acc.numpy()))
    print('Test miou: ' + str(test_miou.numpy()))


    '''
    TODO:
    
    load data differently, with tf.data / tfrecords:
     https://www.tensorflow.org/guide/datasets
     https://www.tensorflow.org/guide/datasets#reading_input_data 
     
     
    Use tf.image for augmentation instead of the imaug library
    
    
    Use @tf.function for speeding up the code
    https://www.tensorflow.org/alpha/tutorials/eager/tf_function
    
    Clean up the code and simply even more
    '''