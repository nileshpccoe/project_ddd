import os 
import sys
import json
import matplotlib.pyplot as plt

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

from cnnClassifier.entity.config_entity import EvaluationConfig



    


class EvaluationModel:
    def __init__(self, config:EvaluationConfig):
        self.config=config

    def save_json(self):
        scores = {
            "Model Name": self.config.model_name,
            "Testing Loss": self.config.hist.history['val_loss'],
            "Testing accuracy": self.config.hist.history['val_accuracy'],
            "Training Loss": self.config.hist.history['loss'],
            "Training Accuracy": self.config.hist.history['accuracy']
        }

        # Ensure the directories exist
        os.makedirs(os.path.join(self.config.folder_path, self.config.model_name), exist_ok=True)

        json_path = os.path.join(self.config.folder_path, self.config.model_name, str(time.time()) + '.json')
        with open(json_path, "w") as f:
            json.dump(scores, f, indent=4)

    def save_score(self):
        # Ensure the directories exist
        os.makedirs(os.path.join(self.config.folder_path, self.config.model_name), exist_ok=True)

        fig_path = os.path.join(self.config.folder_path, self.config.model_name, str(time.time()) + '.png')

        # Create subplots for each metric
        fig, axs = plt.subplots(3, 1, figsize=(5,6))

        # Plot training and validation loss
        axs[0].plot(self.config.hist.history['loss'], label='Training Loss')
        axs[0].plot(self.config.hist.history['val_loss'], label='Validation Loss')
        axs[0].set_title('Training and Validation Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Plot training and validation accuracy
        axs[1].plot(self.config.hist.history['accuracy'], label='Training Accuracy')
        axs[1].plot(self.config.hist.history['val_accuracy'], label='Validation Accuracy')
        axs[1].set_title('Training and Validation Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        # Plot learning rate
        axs[2].plot(self.config.hist.history['lr'], label='Learning Rate', marker='o')
        axs[2].set_title('Learning Rate')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Learning Rate')
        axs[2].legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot
        plt.savefig(fig_path)
        print(self.config.model_name)

        # Show the plot (optional)
        # plt.show()

# Define paths and data elsewhere in your code

     