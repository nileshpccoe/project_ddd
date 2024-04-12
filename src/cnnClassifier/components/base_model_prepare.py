import os 
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dropout, Input, Flatten, Dense, MaxPooling2D

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50, VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dropout, Input, Flatten, Dense, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataclasses import dataclass
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import json
from typing import Iterator
import time
import os
import matplotlib.pyplot as plt
import zipfile
from dataclasses import dataclass
from google_drive_downloader import GoogleDriveDownloader as gdd





class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def getbasemodel(self,pretrained_model):
        try:
            bmodel = pretrained_model
            hmodel = bmodel.output
            hmodel = Flatten()(hmodel)
            hmodel = Dense(64, activation='relu')(hmodel)
            hmodel = Dropout(0.5)(hmodel)
            hmodel = Dense(4, activation='softmax')(hmodel)
            model = Model(inputs=bmodel.input, outputs=hmodel)
            for layer in bmodel.layers:
                layer.trainable = False
        except Exception as e:
            raise e
        return model
    def save_model(self,model):
        try:
            model_saved_path = self.config.model_saved_path
            checkpoint = ModelCheckpoint(model_saved_path, monitor='val_loss', save_best_only=True, verbose=1)
            earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)
            learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)
            callbacks = [checkpoint, earlystop, learning_rate]
            model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

            hist=model.fit(self.config.train_data,steps_per_epoch=self.config.train_data.samples//self.config.params_batchsize,
                              validation_data=self.config.validation_data,
                              validation_steps=self.config.validation_data.samples//self.config.params_batchsize,
                              callbacks=callbacks,
                                epochs=2)
        except Exception as e:
            raise e
        return hist



