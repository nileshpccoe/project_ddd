import os 
import sys
from dataclasses import dataclass


@dataclass
class DataTransformConfig:
    train_path: str
    test_path: str
