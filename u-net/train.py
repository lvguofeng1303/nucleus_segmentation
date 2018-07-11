import os
import numpy as np
from dataset.dataset import NucleusDataset
from model.unet import resnet101_fpn, unet

from losses import make_loss, hard_dice_coef, hard_dice_coef_ch1
from keras import Model, Input
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

train_data_dir = os.path.join(ROOT_DIR, "dataset/stage1_train/")
test_data_dir = os.path.join(ROOT_DIR, "dataset/stage1_test/")

def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index + 1]:
            l.trainable = False

def unet_easy():
    # datatset
    dataset = NucleusDataset()
    dataset.load_nucleus(dataset_dir=train_data_dir)
    dataset.prepare()
    nucleus, masks = dataset.load_dataset()
    print(nucleus.shape, masks.shape)

    # model
    model = unet(input_size=(256, 256, 3), pre_weights=None, channels = 1)
    # model.summary()
    # freeze_model(model, "input_1")
    optimizer = RMSprop(lr=0.001)
    best_model_file = '{}/best_{}.h5'.format("results", "unet")
    checkpointer = ModelCheckpoint('trained_model_weight/model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
    model.compile(optimizer=optimizer, loss=make_loss('bce_dice'),
                      metrics=[binary_crossentropy, hard_dice_coef])
    results = model.fit(nucleus, masks, validation_split=0.1, batch_size=16, epochs=50, 
                    callbacks=[checkpointer])


if __name__ == '__main__':
    unet_easy()