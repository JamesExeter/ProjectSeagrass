from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

import warnings
warnings.filterwarnings('ignore')

"""
Implementation of Inception-Residual Network v1 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.

Some additional details:
[1] Each of the A, B and C blocks have a 'scale_residual' parameter.
    The scale residual parameter is according to the paper. It is however turned OFF by default.

    Simply setting 'scale=True' in the create_inception_resnet_v2() method will add scaling.

[2] There were minor inconsistencies with filter size in both B and C blocks.

    In the B blocks: 'ir_conv' nb of filters  is given as 1154, however input size is 1152.
    This causes inconsistencies in the merge-add mode, therefore the 'ir_conv' filter size
    is reduced to 1152 to match input size.

    In the C blocks: 'ir_conv' nb of filter is given as 2048, however input size is 2144.
    This causes inconsistencies in the merge-add mode, therefore the 'ir_conv' filter size
    is increased to 2144 to match input size.

    Currently trying to find a proper solution with original nb of filters.

[3] In the stem function, the last Convolutional2D layer has 384 filters instead of the original 256.
    This is to correctly match the nb of filters in 'ir_conv' of the next A blocks.
"""

def inception_resnet_stem(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2, 2))(input)
    c = Convolution2D(32, 3, 3, activation='relu', )(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(c)

    c1 = MaxPooling2D((3, 3), strides=(2, 2))(c)
    c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2, 2))(c)

    m = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c1 = Convolution2D(96, 3, 3, activation='relu', )(c1)

    c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same')(m)
    c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same')(c2)
    c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid')(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    p1 = MaxPooling2D((3, 3), strides=(2, 2), )(m2)
    p2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2))(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=channel_axis)
    m3 = BatchNormalization(axis=channel_axis)(m3)
    m3 = Activation('relu')(m3)
    return m3

def inception_resnet_v2_A(input, scale_residual=True):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(ir2)

    ir3 = Convolution2D(32, 1, 1, activation='relu', border_mode='same')(input)
    ir3 = Convolution2D(48, 3, 3, activation='relu', border_mode='same')(ir3)
    ir3 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(ir3)

    ir_merge = merge([ir1, ir2, ir3], concat_axis=channel_axis, mode='concat')

    ir_conv = Convolution2D(384, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out

def inception_resnet_v2_B(input, scale_residual=True):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(128, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(160, 1, 7, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(192, 7, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis)

    ir_conv = Convolution2D(1152, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out

def inception_resnet_v2_C(input, scale_residual=True):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input is relu activation
    init = input

    ir1 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)

    ir2 = Convolution2D(192, 1, 1, activation='relu', border_mode='same')(input)
    ir2 = Convolution2D(224, 1, 3, activation='relu', border_mode='same')(ir2)
    ir2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same')(ir2)

    ir_merge = merge([ir1, ir2], mode='concat', concat_axis=channel_axis)

    ir_conv = Convolution2D(2144, 1, 1, activation='linear', border_mode='same')(ir_merge)
    if scale_residual: ir_conv = Lambda(lambda x: x * 0.1)(ir_conv)

    out = merge([init, ir_conv], mode='sum')
    out = BatchNormalization(axis=channel_axis)(out)
    out = Activation("relu")(out)
    return out


def reduction_A(input, k=192, l=224, m=256, n=384):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2))(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same')(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2))(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=1)(m)
    m = Activation('relu')(m)
    return m


def reduction_resnet_v2_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    r2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r2 = Convolution2D(384, 3, 3, activation='relu', subsample=(2,2))(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r3 = Convolution2D(288, 3, 3, activation='relu', subsample=(2, 2))(r3)

    r4 = Convolution2D(256, 1, 1, activation='relu', border_mode='same')(input)
    r4 = Convolution2D(288, 3, 3, activation='relu', border_mode='same')(r4)
    r4 = Convolution2D(320, 3, 3, activation='relu', subsample=(2, 2))(r4)

    m = merge([r1, r2, r3, r4], concat_axis=channel_axis, mode='concat')
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m

def create_inception_resnet_v2(nb_classes=1001, scale=True):
    '''
    Creates a inception resnet v2 network

    :param nb_classes: number of classes.txt
    :param scale: flag to add scaling of activations
    :return: Keras Model with 1 input (299x299x3) input shape and 2 outputs (final_output, auxiliary_output)
    '''

    if K.image_dim_ordering() == 'th':
        init = Input((3, 299, 299))
    else:
        init = Input((299, 299, 3))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_resnet_stem(init)

    # 10 x Inception Resnet A
    for i in range(10):
        x = inception_resnet_v2_A(x, scale_residual=scale)

    # Reduction A
    x = reduction_A(x, k=256, l=256, m=384, n=384)

    # 20 x Inception Resnet B
    for i in range(20):
        x = inception_resnet_v2_B(x, scale_residual=scale)

    # Auxiliary tower
    aux_out = AveragePooling2D((5, 5), strides=(3, 3))(x)
    aux_out = Convolution2D(128, 1, 1, border_mode='same', activation='relu')(aux_out)
    aux_out = Convolution2D(768, 5, 5, activation='relu')(aux_out)
    aux_out = Flatten()(aux_out)
    aux_out = Dense(nb_classes, activation='softmax')(aux_out)

    # Reduction Resnet B
    x = reduction_resnet_v2_B(x)

    # 10 x Inception Resnet C
    for i in range(10):
        x = inception_resnet_v2_C(x, scale_residual=scale)

    # Average Pooling
    x = AveragePooling2D((8,8))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, output=[out, aux_out], name='Inception-Resnet-v2')
    return model

if __name__ == "__main__":
    from keras.utils.visualize_util import plot

    inception_resnet_v2 = create_inception_resnet_v2()
    #inception_resnet_v2.summary()

    plot(inception_resnet_v2, to_file="Inception ResNet-v2.png", show_shapes=True)