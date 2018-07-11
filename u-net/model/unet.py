# -*- coding: utf-8 -*-
"""
This file have implement two U-Net model
    - The first one is the original U-Net network, but our model in conv2d use padding='same'
    - The second is the unet which use Resnet101
"""
from keras import Model, Input
from keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation, concatenate, Add, MaxPooling2D, Dropout
from keras.utils import get_file
from keras.applications.resnet50 import ResNet50

from resnets import ResNet101

resnet_filename = 'ResNet-{}-model.keras.h5'
resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
"""
def download_resnet_imagenet(v):
    v = int(v.replace('resnet', ''))

    filename = resnet_filename.format(v)
    resource = resnet_resource.format(v)
    if v == 50:
        checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
    elif v == 101:
        checksum = '05dc86924389e5b401a9ea0348a3213c'
    elif v == 152:
        checksum = '6ee11ef2b135592f8031058820bb9e71'

    return get_file(
        filename,
        resource,
        cache_subdir='models',
        md5_hash=checksum
    )
"""

def unet(input_size, pre_weights, channels, name="unet"):
    inputs = Input(input_size)

    # down convolution stage
    down1 = conv_bn_relu_twice(inputs, 64, kernel_size=3, name=name+"_down1")
    down1_pool = max_pooling(down1, name=name+"_pool1")
    down2 = conv_bn_relu_twice(down1_pool, 128, kernel_size=3, name=name+"_down2")
    down2_pool = max_pooling(down2, name=name+"_pool2")
    down3 = conv_bn_relu_twice(down2_pool, 256, kernel_size=3, name=name+"_down3")
    down3_pool = max_pooling(down3, name=name+"_pool3")

    # down4 add Dropout layer
    down4 = conv_bn_relu_twice(down3_pool, 512, kernel_size=3, name=name+"_down4")
    drop4 = dropout(down4, name=name+"_drop4")
    down4_pool = max_pooling(drop4, name=name+"_pool4")
    # down5 add Dropout layer
    down5 = conv_bn_relu_twice(down4_pool, 1024, kernel_size=3, name=name+"_down5")
    drop5 = dropout(down5, name=name+"_drop5")

    # up convolution stage
    up4 = upconv_bn(input=drop5, down_input=drop4, filter=512, name=name+"_up4")
    up3 = upconv_bn(input=up4, down_input=down3, filter=256, name=name+"_up3")
    up2 = upconv_bn(input=up3, down_input=down2, filter=128, name=name+"_up2")
    up1 = upconv_bn(input=up2, down_input=down1, filter=64, name=name+"_up1")

    output = Conv2D(channels, (1, 1), activation='sigmoid', name=name+"_output")(up1)
    model = Model(inputs, output)
    return model

def resnet101_fpn(input_size, pre_weights, channels=1, activation="softmax"):
    inputs = Input(input_size)
    resnet101_base = ResNet101(inputs)
    #resnet101_base.load_weights(download_resnet_imagenet("resnet101"))
    conv1 = resnet101_base.get_layer("conv1_relu").output
    conv2 = resnet101_base.get_layer("res2c_relu").output
    conv3 = resnet101_base.get_layer("res3b3_relu").output
    conv4 = resnet101_base.get_layer("res4b22_relu").output
    conv5 = resnet101_base.get_layer("res5c_relu").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2")
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1,1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    if activation == 'softmax':
        name = 'mask_softmax'
        x = Conv2D(channels, (1, 1), activation=activation, name=name)(x)
    else:
        x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(inputs, x)
    return model

def decoder_block_no_bn(input, filters, skip, block_name, activation='relu'):
    x = UpSampling2D()(input)
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation)
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation)
    return x


def prediction_fpn_block(x, name, upsample=None, feature_size=128):
    # 1
    x = Conv2D(filters=feature_size, kernel_size=(3,3), strides=(1,1), padding='same', 
                kernel_initializer="he_normal", name="prediction_"+name+"_1")(x)
    x = Activation("relu", name="prediction_relu_"+name+"_1")(x)
    # 2
    x = Conv2D(filters=feature_size, kernel_size=(3,3), strides=(1,1), padding='same', 
                kernel_initializer="he_normal", name="prediction_"+name+"_2")(x)
    x = Activation("relu", name="prediction_relu_"+name+"_2")(x)
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x

def create_pyramid_features(c1, c2, c3, c4, c5, feature_size=256):
    P5 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1), 
                    padding='same', name='P5', kernel_initializer="he_normal" )(c5)
    P5_upsampled = UpSampling2D((2, 2), name="P5_upsampled")(P5)

    P4 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='C4_reduce', kernel_initializer="he_normal")(c4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='P4', kernel_initializer="he_normal")(P4)
    P4_upsampled = UpSampling2D((2, 2), name="P4_upsampled")(P4)

    P3 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='C3_reduce', kernel_initializer="he_normal")(c3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='P3', kernel_initializer="he_normal")(P3)
    P3_upsampled = UpSampling2D((2, 2), name="P3_upsampled")(P3)

    P2 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='C2_reduce', kernel_initializer="he_normal")(c2)
    P2 = Add(name='P2_merged')([P3_upsampled, P2])
    P2 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='P2', kernel_initializer="he_normal")(P2)
    P2_upsampled = UpSampling2D((2, 2), name="P2_upsampled")(P2)

    P1 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='C1_reduce', kernel_initializer="he_normal")(c1)
    P1 = Add(name='P1_merged')([P2_upsampled, P1])
    P1 = Conv2D(filters=feature_size, kernel_size=(1, 1), strides=(1,1),
                    padding='same', name='P1', kernel_initializer="he_normal")(P1)
    
    return P1, P2, P3, P4, P5

def conv_relu(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu'):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = Activation(activation, name=name + '_relu')(x)
    return x

def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99,
                 bn_scale=True, use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def conv_bn_relu_twice(input, filter, name, kernel_size=3, stride=(1, 1), padding='same', bn_axis=-1, bn_momentum=0.99,bn_scale=True, use_bias=True):
    # first conv2d
    x = Conv2D(filters=filter, kernel_size=(kernel_size, kernel_size), 
                strides=stride, padding=padding, use_bias=use_bias, kernel_initializer="he_normal", name=name+"_conv1")(input)
    x = BatchNormalization(scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, name=name+"_bn1")(x)
    x = Activation('relu', name=name+"_relu1")(x)
    
    # second conv2d
    x = Conv2D(filters=filter, kernel_size=(kernel_size, kernel_size), 
                strides=stride, padding=padding, use_bias=use_bias, kernel_initializer="he_normal", name=name+"_conv2")(x)
    x = BatchNormalization(scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, name=name+"_bn2")(x)
    x = Activation('relu', name=name+"_relu2")(x)
    return x

def max_pooling(input, name, pool_size=(2, 2)):
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name=name)(input)
    return x

def dropout(input, name, droprate=0.5):
    x = Dropout(rate=0.5, name=name)(input)
    return x


def upconv_bn(input, down_input, filter, name, kernel_size=3, strides=(1, 1), padding='same', up_size=(2,2), use_bias=True):
    x = UpSampling2D(up_size)(input)
    x = Conv2D(filters=filter, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=use_bias, kernel_initializer="he_normal",
                name=name+"_conv0")(x)
    x = concatenate([x, down_input], axis=-1, name=name+"_concat")
    x = conv_bn_relu_twice(x, filter, kernel_size=3, name=name)
    return x





if __name__ == '__main__':
    #unet(input_size=(256, 256, 1)).summary()
    a = Input((256, 256, 1))
    model = resnet101_fpn(input_size=(256, 256, 3), pre_weights=None)
    model.summary()
