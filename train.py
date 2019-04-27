import numpy as np
import tensorflow as tf
import nets.MiniNetv2 as MiniNetv2
import os
import utils.Loader as Loader
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, apply_augmentation, get_metrics
tf.random.set_seed(7)
np.random.seed(7)

# @tf.function
def train_step(model, x, y, mask, loss_function, optimizer, labels_resize_factor, size_input):
    with tf.GradientTape() as tape:

        [x, y, mask] = convert_to_tensors([x, y, mask]) # convert numpy to tensor
        mask = tf.expand_dims(mask, axis=-1)

        x, y, mask = apply_augmentation(x, y, mask, size_input)

        #resize if needed
        if labels_resize_factor!= 1:
            mask = tf.image.resize(mask, (int(mask.shape[1]/labels_resize_factor), int(mask.shape[2]/labels_resize_factor)), method=tf.image.ResizeMethod.BILINEAR)
            y = tf.image.resize(y, (int(y.shape[1]/labels_resize_factor), int(y.shape[2]/labels_resize_factor)), method=tf.image.ResizeMethod.BILINEAR)


        y_ = model(x, training=True)  # get output of the model.

        # Apply mask to ignore labels
        y *= mask

        loss = loss_function(y, y_) # apply loss


    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))# Apply gradientes

    return loss



# Trains the model for certains epochs on a dataset
def train(loader, optimizer, loss_function, model, size_input,  epochs=5, batch_size=2, lr=None, init_lr=2e-4,
          evaluation=True, name_best_model = 'weights/best', preprocess_mode=None, labels_resize_factor=1):
    # Parameters for training
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / batch_size) + 1
    best_miou = 0
    log_freq = min(50, int(steps_per_epoch/5))
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train') # tensorboard
    test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test') # tensorboard
    print('Please enter in terminal: tensorboard --logdir \\tmp\\summaries')

    for epoch in range(epochs):  # for each epoch
        lr_decay(lr, init_lr, 1e-9, epoch, epochs - 1)  # compute the new lr
        print('epoch: ' + str(epoch+1) + '. Learning rate: ' + str(lr.numpy()))
        for step in range(steps_per_epoch):  # for every batch
            # get batch
            x, y, mask = loader.get_batch(size=batch_size, train=True)
            x = preprocess(x, mode=preprocess_mode)

            with train_summary_writer.as_default():
                loss = train_step(model, x, y, mask, loss_function, optimizer, labels_resize_factor, size_input)
                # tensorboard
                avg_loss.update_state(loss)
                if tf.equal(optimizer.iterations % log_freq, 0):
                    tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
                    avg_loss.reset_states()


        if evaluation:
            # get metrics
            with train_summary_writer.as_default():
                train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode,
                                                    labels_resize_factor=labels_resize_factor, optimizer=optimizer)

            with test_summary_writer.as_default():
                test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,
                                                  scales=[1], preprocess_mode=preprocess_mode, labels_resize_factor=labels_resize_factor, optimizer=optimizer)

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
    width_test = 480
    height_test = 360
    width_train = int(width_test/2) # will be cropped from width_test size
    height_train = int(height_test/2)  # will be cropped from height_test size



    labels_resize_factor = 2 # factor to divide the label size (must matach with the output size of the CNN
    channels = 3
    lr = 8e-4
    name_best_model = os.path.join('weights', 'camvid', 'best')
    dataset_path = os.path.join('Datasets', 'camvid')
    preprocess_mode = 'imagenet'  #possible values 'imagenet', 'normalize',None

    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation',
                           width=width_test, height=height_test, channels=channels)

    # build model
    model = MiniNetv2.MiniNetv2(num_classes=n_classes)
    #model = Net.Segception(n_classes, input_shape=(None, None, 3), weights='imagenet')

    # optimizer
    learning_rate = tf.Variable(lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_function = tf.losses.CategoricalCrossentropy()

    # restore if model saved and show number of params
    restore_state(model, name_best_model)



    train(loader=loader, optimizer=optimizer, loss_function=loss_function, model=model, size_input=(height_train, width_train), epochs=epochs, batch_size=batch_size,
          lr=learning_rate, init_lr=lr,  name_best_model=name_best_model, evaluation=True, preprocess_mode=preprocess_mode,
          labels_resize_factor=labels_resize_factor)

    get_params(model)

    print('Testing model')
    test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1],
                                      write_images=True, preprocess_mode=preprocess_mode, time_exect=False, labels_resize_factor=labels_resize_factor)
    print('Test accuracy: ' + str(test_acc.numpy()))
    print('Test miou: ' + str(test_miou.numpy()))


