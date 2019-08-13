from keras.applications import VGG16
from keras.layers import Lambda, Dense
from keras.optimizers import SGD
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.utils import np_utils
import theano.tensor.nnet.abstract_conv as absconv
import os
import glob
import numpy as np
import cv2
import argparse


def get_classmap(model, X, nb_classes, batch_size, num_input_channels, ratio):

    inc = model.layers[0].input
    conv6 = model.layers[-4].output
    conv6_resized = absconv.bilinear_upsampling(conv6,
                                                ratio,
                                                batch_size=batch_size,
                                                num_input_channels=num_input_channels)
    WT = model.layers[-1].W.T
    conv6_resized = K.reshape(conv6_resized,
                              (-1, num_input_channels, 224 * 224))
    classmap = K.dot(WT, conv6_resized).reshape((-1, nb_classes, 224, 224))
    get_cmap = K.function([inc], classmap)
    return get_cmap([X])


def train_VGGCAM(VGG_weight_path, nb_classes, num_input_channels=1024):
    """
    Train VGGCAM model
    args: VGG_weight_path (str) path to keras vgg16 weights
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer
    """

    # Load model
    model = VGGCAM(nb_classes)

    # Load weights
    with h5py.File(VGG_weight_path) as hw:
        for k in range(hw.attrs['nb_layers']):
            g = hw['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            model.layers[k].set_weights(weights)
            if model.layers[k].name == "convolution2d_13":
                break
        print('Model loaded.')

    # Compile
    model.compile(optimizer="sgd", loss='categorical_crossentropy')

    # Train model with your data (dummy code)
    # update with your data

    # N.B. The data should be compatible with the VGG16 model style:

    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)

    # model.fit(X, y)

    # Save model
    model.save_weights(os.path.join('%s_weights.h5' % model.name))


def plot_classmap(VGGCAM_weight_path, img_path, label,
                  nb_classes, num_input_channels=1024, ratio=16):
    """
    Plot class activation map of trained VGGCAM model
    args: VGGCAM_weight_path (str) path to trained keras VGGCAM weights
          img_path (str) path to the image for which we get the activation map
          label (int) label (0 to nb_classes-1) of the class activation map to plot
          nb_classes (int) number of classes
          num_input_channels (int) number of conv filters to add
                                   in before the GAP layer
          ratio (int) upsampling ratio (16 * 14 = 224)
    """

    # Load and compile model
    model = VGGCAM(nb_classes, num_input_channels)
    model.load_weights(VGGCAM_weight_path)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    # Load and format data
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    # Get a copy of the original image
    im_ori = im.copy().astype(np.uint8)
    # VGG model normalisations
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))

    batch_size = 1
    classmap = get_classmap(model,
                            im.reshape(1, 3, 224, 224),
                            nb_classes,
                            batch_size,
                            num_input_channels=num_input_channels,
                            ratio=ratio)

    plt.imshow(im_ori)
    plt.imshow(classmap[0, label, :, :],
               cmap="jet",
               alpha=0.5,
               interpolation='nearest')
    plt.show()
    raw_input()


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def get_model():
    model = VGG16(include_top=False, weights='imagenet',
                  input_shape=(256, 256, 3))
    # model = VGG16_convolutions()

    # model = load_model_weights(model, "vgg16_weights.h5")

    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))
    model.add(Dense(10, activation='softmax', init='uniform'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def load_inria_person(path):
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "/neg")
    pos_images = [cv2.resize(cv2.imread(x),
                             (64, 128)) for x in glob.glob(pos_path +
                                                           "/*.png")]
    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images]
    neg_images = [cv2.resize(cv2.imread(x),
                             (64, 128)) for x in glob.glob(neg_path +
                                                           "/*.png")]
    neg_images = [np.transpose(img, (2, 0, 1)) for img in neg_images]
    y = [1] * len(pos_images) + [0] * len(neg_images)
    y = np_utils.to_categorical(y, 2)
    X = np.float32(pos_images + neg_images)

    return X, y


def train(dataset_path):
        model = get_model()
        X, y = load_inria_person(dataset_path)
        print("Training..")
        checkpoint_path = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                     verbose=0, save_best_only=False,
                                     save_weights_only=False, mode='auto')
        model.fit(X, y, nb_epoch=40, batch_size=32, validation_split=0.2,
                  verbose=1, callbacks=[checkpoint])


def visualize_class_activation_map(model_path, img_path, output_path):
        model = load_model(model_path)
        original_img = cv2.imread(img_path, 1)
        width, height, _ = original_img.shape

        # Reshape to the network input shape (3, w, h).
        img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

        # Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = model.get_output_layer(model, "conv5_3")
        get_output = K.function([model.layers[0].input],
                                [final_conv_layer.output,
                                 model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        # Create the class activation map.
        cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
        for i, w in enumerate(class_weights[:, 1]):
                cam += w * conv_outputs[i, :, :]
        print("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img
        cv2.imwrite(output_path, img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False,
                        help='Train the network or visualize a CAM')
    parser.add_argument("--image_path", type=str,
                        help="Path of an image to run the network on")
    parser.add_argument("--output_path", type=str, default="heatmap.jpg",
                        help="Path of an image to run the network on")
    parser.add_argument("--model_path", type=str,
                        help="Path of the trained model")
    parser.add_argument("--dataset_path", type=str,
                        help="Path to image dataset."
                        " Should have pos/neg folders, like in the"
                        "inria person dataset."
                        "http://pascal.inrialpes.fr/data/human/")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    if args.train:
        train(args.dataset_path)
    else:
        visualize_class_activation_map(args.model_path,
                                       args.image_path,
                                       args.output_path)
