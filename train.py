import numpy as np
import tensorflow as tf
import nets.MiniNetv2 as MiniNetv2
import nets.ResNet50 as ResNet50
import nets.EfficientNet as EfficientNet
import os
import utils.Loader as Loader
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, apply_augmentation, get_metrics,init_model
import argparse
import time
import sys
import cv2

np.random.seed(7)
tf.random.set_seed(7)

@tf.function
def train_step(model, x, y, mask, loss_function, optimizer, size_input, zoom_factor):
    with tf.GradientTape() as tape:

        [x, y, mask] = convert_to_tensors([x, y, mask]) # convert numpy to tensor
        mask = tf.expand_dims(mask, axis=-1)

        x, y, mask = apply_augmentation(x, y, mask, size_input, zoom_factor)

        #resize if needed
        # if label_resize_factor != 1:
        #     mask = tf.image.resize(mask, (int(mask.shape[1]/crop_factor), int(mask.shape[2]/crop_factor)), method=tf.image.ResizeMethod.BILINEAR)
        #     y = tf.image.resize(y, (int(y.shape[1]/crop_factor), int(y.shape[2]/crop_factor)), method=tf.image.ResizeMethod.BILINEAR)

        y_ = model(x, training=True)  # get output of the model.

        # Apply mask to ignore labels
        y *= mask
        loss = loss_function(y, y_) # apply loss
        # print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))# Apply gradientes

    return loss



# Trains the model for certains epochs on a dataset
def train(loader, optimizer, loss_function, model, config=None, lr=None,
          evaluation=True, name_best_model='weights/best', preprocess_mode=None):
    # Parameters for training
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / config['batch_size']) + 1
    best_miou = 0
    log_freq = min(50, int(steps_per_epoch/5))
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train') # tensorboard
    test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test') # tensorboard
    print('Please enter in terminal: tensorboard --logdir /tmp/summaries')

    for epoch in range(config['epochs']):  # for each epoch
        start_time_epoch = time.time()
        lr_decay(lr, config['init_lr'], 1e-9, epoch, config['epochs'] - 1)  # compute the new lr
        print('epoch: ' + str(epoch+1) + '. Learning rate: ' + str(lr.numpy()))

        for step in range(steps_per_epoch):  # for every batch

            # get batch
            x, y, mask = loader.get_batch(size=config['batch_size'], train=True)

            x = preprocess(x, mode=preprocess_mode)

            with train_summary_writer.as_default():
                loss = train_step(model, x, y, mask, loss_function, optimizer, (config['height_train'], config['width_train']), config['zoom_augmentation'])
                # tensorboard
                avg_loss.update_state(loss)
                if tf.equal(optimizer.iterations % log_freq, 0):
                    tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                    avg_loss.reset_states()

        if evaluation:
            # get metrics

            # with train_summary_writer.as_default():
            #     train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, flip_inference=False, preprocess_mode=preprocess_mode, optimizer=optimizer)

            with test_summary_writer.as_default():
                test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False, preprocess_mode=preprocess_mode, optimizer=optimizer, scales=[1])

            # print('Train accuracy: ' + str(train_acc.numpy()))
            # print('Train miou: ' + str(train_miou.numpy()))
            print('Test accuracy: ' + str(test_acc.numpy()))
            print('Test miou: ' + str(test_miou.numpy()))

            # save model if best model
            if test_miou.numpy() > best_miou:
                best_miou = test_miou.numpy()
                model.save_weights(name_best_model)

            print('Current Best model miou: ' + str(best_miou))
            print('')

        else:
            model.save_weights(name_best_model)

        loader.suffle_segmentation()  # sheffle training set every epoch
        print('Epoch time seconds: ' + str(time.time()-start_time_epoch))

if __name__ == "__main__":

    CONFIG = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", help="Number of classes to classify", default=11)
    parser.add_argument("--batch_size", help="Number of samples per batch", default=4)
    parser.add_argument("--init_lr", help="Initial learning rate", default=1e-4)
    parser.add_argument("--epochs", help="Number of epochs to train the dataset", default=400)
    parser.add_argument("--median_frequency", help="Factor to power the median frequency balancing (0 to none effect, 1 to full efect)", default=0.15)
    parser.add_argument("--width", help="width of the dataset images", default=960)
    parser.add_argument("--height", help="height of the dataset images", default=720)
    parser.add_argument("--train_crop_divide_factor_x", help="Factor to divide the images resolution when training (x_axis)", default=2)
    parser.add_argument("--train_crop_divide_factor_y", help="Factor to divide the images resolution when training (y_axis)", default=1.25)
    parser.add_argument("--zoom_augmentation", help="Factor to zoom in or zoom out during augmentation, i.e., zoom (1+factor, 1-factor)", default=0.2)
    parser.add_argument("--weights_path", help="Path to the model weights", default='weights/camvid_resnet/model')
    parser.add_argument("--dataset_path", help="Path to the dataset", default='datasets/camvid')
    parser.add_argument("--preprocess", help="Preprocess of the dataset", choices=['imagenet', 'normalize', None], default='imagenet')
    args = parser.parse_args()


    CONFIG['n_classes'] = int(args.n_classes)
    CONFIG['batch_size'] = int(args.batch_size)
    CONFIG['epochs'] = int(args.epochs)
    CONFIG['width'] = int(args.width)
    CONFIG['height'] = int(args.height)
    CONFIG['crop_factor_x'] = float(args.train_crop_divide_factor_x)
    CONFIG['crop_factor_y'] = float(args.train_crop_divide_factor_y)
    CONFIG['width_train'] = int(CONFIG['width'] / CONFIG['crop_factor_x']) # will be cropped from width_test size
    CONFIG['height_train']  = int(CONFIG['height'] / CONFIG['crop_factor_y'])  # will be cropped from height_test size
    CONFIG['init_lr'] = float(args.init_lr)
    CONFIG['median_frequency'] = float(args.median_frequency)
    CONFIG['zoom_augmentation'] = float(args.zoom_augmentation)

    assert CONFIG['width'] * (1 - CONFIG['zoom_augmentation'] ) >= CONFIG['width_train']
    assert CONFIG['height'] * (1 - CONFIG['zoom_augmentation'] ) >= CONFIG['height_train']


    # GPU to use
    n_gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)

    # Loader
    loader = Loader.Loader(dataFolderPath=args.dataset_path, n_classes=CONFIG['n_classes'], width=CONFIG['width'], height=CONFIG['height'], median_frequency=CONFIG['median_frequency'])

    # build model
    #model = MiniNetv2.MiniNetv2p(num_classes=CONFIG['n_classes'])
    model = ResNet50.ResNet50Seg(CONFIG['n_classes'], input_shape=(None, None, 3), weights='imagenet')

    # optimizer
    learning_rate = tf.Variable(CONFIG['init_lr'])
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    # restore if model saved and show number of params
    restore_state(model, args.weights_path)

    init_model(model, (1, CONFIG['width'], CONFIG['height'], 3))
    get_params(model)


    # Train
    train(loader=loader, optimizer=optimizer, loss_function=loss_function, model=model, config=CONFIG,
          lr=learning_rate,  name_best_model=args.weights_path, evaluation=True, preprocess_mode=args.preprocess)


    print('Testing model')
    test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1, 2, 1.5, 0.5, 0.75],
                                      write_images=True, preprocess_mode=args.preprocess, time_exect=True)
    print('Test accuracy: ' + str(test_acc.numpy()))
    print('Test miou: ' + str(test_miou.numpy()))


