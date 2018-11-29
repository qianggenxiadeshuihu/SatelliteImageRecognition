from keras.models import Model
from keras.layers import concatenate
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam

from .metrics import jaccard_coef, jaccard_coef_int


def get_unet(feature_count=3):
    conv_params = dict(activation='relu', border_mode='same', data_format='channels_last')
    pool_params = dict(data_format='channels_last')
    inputs = Input((256, 256, feature_count))
    conv1 = Convolution2D(32, (3, 3), **conv_params)(inputs)
    conv1 = Convolution2D(32, (3, 3), **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), **pool_params)(conv1)

    conv2 = Convolution2D(64, (3, 3), **conv_params)(pool1)
    conv2 = Convolution2D(64, (3, 3), **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), **pool_params)(conv2)

    conv3 = Convolution2D(128, (3, 3), **conv_params)(pool2)
    conv3 = Convolution2D(128, (3, 3), **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), **pool_params)(conv3)

    conv4 = Convolution2D(256, (3, 3), **conv_params)(pool3)
    conv4 = Convolution2D(256, (3, 3), **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), **pool_params)(conv4)

    conv5 = Convolution2D(512, (3, 3), **conv_params)(pool4)
    conv5 = Convolution2D(512, (3, 3), **conv_params)(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2), **pool_params)(conv5), conv4], axis=-1)
    conv6 = Convolution2D(256, (3, 3), **conv_params)(up6)
    conv6 = Convolution2D(256, (3, 3), **conv_params)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2), **pool_params)(conv6), conv3], axis=-1)
    conv7 = Convolution2D(128, (3, 3), **conv_params)(up7)
    conv7 = Convolution2D(128, (3, 3), **conv_params)(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2), **pool_params)(conv7), conv2], axis=-1)
    conv8 = Convolution2D(64, (3, 3), **conv_params)(up8)
    conv8 = Convolution2D(64, (3, 3), **conv_params)(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2), **pool_params)(conv8), conv1], axis=-1)
    conv9 = Convolution2D(32, (3, 3), **conv_params)(up9)
    conv9 = Convolution2D(32, (3, 3), **conv_params)(conv9)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid', data_format='channels_last')(conv9)
    adam = Adam()

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model
