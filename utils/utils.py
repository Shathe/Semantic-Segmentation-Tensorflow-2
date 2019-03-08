import numpy as np
import tensorflow as tf
import math
import os
import cv2
import time

# Prints the number of parameters of a model
def get_params(model):
    # Init models (variables and input shape)
    total_parameters = 0
    for variable in model.variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print("Total parameters of the net: " + str(total_parameters) + " == " + str(total_parameters / 1000000.0) + "M")

# preprocess a batch of images
def preprocess(x, mode='imagenet'):
    if mode:
        if 'imagenet' in mode:
            return tf.keras.applications.xception.preprocess_input(x)
        elif 'normalize' in mode:
            return  x.astype(np.float32) / 127.5 - 1
    else:
        return x

# applies to a lerarning rate tensor (lr) a decay schedule, the polynomial decay
def lr_decay(lr, init_learning_rate, end_learning_rate, epoch, total_epochs, power=0.9):
    lr.assign(
        (init_learning_rate - end_learning_rate) * math.pow(1 - epoch / 1. / total_epochs, power) + end_learning_rate)

# converts a list of arrays into a list of tensors
def convert_to_tensors(list_to_convert):
    if list_to_convert != []:
        return [tf.convert_to_tensor(list_to_convert[0])] + convert_to_tensors(list_to_convert[1:])
    else:
        return []

# restores a checkpoint model
def restore_state(model, checkpoint):
    try:
        model.load_weights(checkpoint)
        print('Model loaded')
    except Exception as e:
        print('Model not loaded: ' + str(e))

# inits a models (set input)
def init_model(model, input_shape):
    model._set_inputs(np.zeros(input_shape))


# Erase the elements if they are from ignore class. returns the labesl and predictions with no ignore labels
def erase_ignore_pixels(labels, predictions, mask):
    indices = tf.squeeze(tf.where(tf.greater(mask, 0)))  # not ignore labels
    labels = tf.cast(tf.gather(labels, indices), tf.int64)
    predictions = tf.gather(predictions, indices)

    return labels, predictions

# generate and write an image into the disk

def generate_image(image_scores, output_dir, dataset, loader, train=False):
    # Get image name
    if train:
        list = loader.image_train_list
        index = loader.index_train
    else:
        list = loader.image_test_list
        index = loader.index_test

    dataset_name = dataset.split('/')
    if dataset_name[-1] != '':
        dataset_name = dataset_name[-1]
    else:
        dataset_name = dataset_name[-2]

    # Get output dir name
    out_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write it
    image = np.argmax(image_scores, 2)
    name_split = list[index - 1].split('/')
    name = name_split[-1].replace('.jpg', '.png').replace('.jpeg', '.png')
    cv2.imwrite(os.path.join(out_dir, name), image)


def inference(model, x, y, n_classes, flip_inference=True, scales=[1], preprocess_mode=None, time_exect=False):
    x = preprocess(x, mode=preprocess_mode)
    [x] = convert_to_tensors([x])

    # creates the variable to store the scores
    y_ = convert_to_tensors([np.zeros((y.shape[0], y.shape[1], y.shape[2], n_classes), dtype=np.float32)])[0]

    for scale in scales:
        # scale the image
        x_scaled = tf.image.resize(x, (x.shape[1] * scale, x.shape[2] * scale),
                                              method=tf.image.ResizeMethod.BILINEAR)

        pre = time.time()

        y_scaled = model(x_scaled)
        if time_exect and scale == 1:
            print("seconds to inference: " + str((time.time()-pre)*1000) + " ms")

        #  rescale the output
        y_scaled = tf.image.resize(y_scaled, (y.shape[1], y.shape[2]),
                                              method=tf.image.ResizeMethod.BILINEAR)
        # get scores
        y_scaled = tf.nn.softmax(y_scaled)

        if flip_inference:
            # calculates flipped scores
            y_flipped_ = tf.image.flip_left_right(model(tf.image.flip_left_right(x_scaled)))
            # resize to rela scale
            y_flipped_ = tf.image.resize(y_flipped_, (y.shape[1], y.shape[2]), method=tf.image.ResizeMethod.BILINEAR)
            # get scores
            y_flipped_score = tf.nn.softmax(y_flipped_)

            y_scaled += y_flipped_score

        y_ += y_scaled

    return y_

# get accuracy and miou from a model
def get_metrics(loader, model, n_classes, train=True, flip_inference=False, scales=[1], write_images=False, preprocess_mode=None, time_exect=False, labels_resize_factor=1):
    if train:
        loader.index_train = 0
    else:
        loader.index_test = 0

    accuracy = tf.metrics.Accuracy()
    mIoU = tf.metrics.MeanIoU(num_classes=n_classes)

    if train:
        samples = len(loader.image_train_list)
    else:
        samples = len(loader.image_test_list)

    for step in range(samples):  # for every batch
        x, y, mask = loader.get_batch(size=1, train=train, augmenter=False, labels_resize_factor=labels_resize_factor)
        [y] = convert_to_tensors([y])

        y_ = inference(model, x, y, n_classes, flip_inference, scales, preprocess_mode=preprocess_mode, time_exect=time_exect)


        # generate images
        if write_images:
            generate_image(y_[0,:,:,:], 'images_out', loader.dataFolderPath, loader, train)

        # Rephape
        y = tf.reshape(y, [y.shape[1] * y.shape[2] * y.shape[0], y.shape[3]])
        y_ = tf.reshape(y_, [y_.shape[1] * y_.shape[2] * y_.shape[0], y_.shape[3]])
        mask = tf.reshape(mask, [mask.shape[1] * mask.shape[2] * mask.shape[0]])

        labels, predictions = erase_ignore_pixels(labels=tf.argmax(y, 1), predictions=tf.argmax(y_, 1), mask=mask)
        accuracy.update_state(labels, predictions)
        mIoU.update_state(labels, predictions)

    # get the train and test accuracy from the model
    return accuracy.result(), mIoU.result()

