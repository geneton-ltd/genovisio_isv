import os

SRC_DIR = os.path.abspath(os.path.dirname(__file__))

PACKAGE_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PACKAGE_DIR, "models")

HAPLOINSUFFICIENCY_SCORES = [3]
TRIPLOSENSITIVITY_SCORES = [3]
