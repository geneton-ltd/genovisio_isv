import os

SRC_DIR = os.path.abspath(os.path.dirname(__file__))

PACKAGE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PACKAGE_DIR, "models")
TRAIN_DATASET_DIR = os.path.join(PACKAGE_DIR, "datasets")

HAPLOINSUFFICIENCY_SCORES = [1, 2, 3] # zmenit len na 3
TRIPLOSENSITIVITY_SCORES = [1, 2, 3] # zmenit len na 3
