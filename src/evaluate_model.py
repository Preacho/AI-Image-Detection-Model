import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import load_data
import sys

TRAIN_DIR  = "preprocessed_dataset/train"
TEST_DIR   = "preprocessed_dataset/test"
MODELS_DIR = "models"
OUTPUT_XLSX = "results.xlsx"

IMG_SIZE   = (300, 300)
COLOR_MODE = "grayscale"
RANDOM_STATE = 42

CLIP_TRAIN_CACHE = "clip_train.npz"
CLIP_TEST_CACHE  = "clip_test.npz"




if __name__ == "__main__":

    if(len(sys.argv) > 1):
        evaluate_model(sys.argv[1])
    else:   
        evaluate_model()