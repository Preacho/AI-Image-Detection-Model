import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------- CONFIGURATION ----------
PREPROCESSED_DIR = "preprocessed_dataset"
IMG_WIDTH = 300
IMG_HEIGHT = 300
RANDOM_STATE = 42

# ---------- 1. LOAD IMAGES AND LABELS ----------
def load_data_from_npz(npz_path=NPZ_PATH):
    """Load train/test arrays from a .npz file."""
    print(f"Loading data from: {npz_path}")
    data = np.load(npz_path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # Flatten images if they are still (N, H, W, C) or (N, H, W)
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test  = X_test.reshape(X_test.shape[0], -1)

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    class_names = [str(c) for c in np.unique(y_train)]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Classes    : {class_names}")
    print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    return X_train, X_test, y_train, y_test, class_names


def load_data():
    """Load train and test data from preprocessed directories"""
    
    train_dir = os.path.join(PREPROCESSED_DIR, "train")
    test_dir = os.path.join(PREPROCESSED_DIR, "test")
    
    X_train, y_train, class_names = load_data_from_npz("training_data.npz")
    X_test, y_test, _ = load_data_from_npz("testing_data.npz")
    
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, class_names

def train_random_forest():
    """Main training pipeline for Random Forest"""
    
    # Load data using npz
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    X_train, X_test, y_train, y_test, class_names = load_data()
    
    
    print("\n" + "="*60)
    print("FEATURE SCALING")
    print("="*60)

    use_scaling = False  
    
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
        joblib.dump(scaler, "models/scaler.pkl")
    else:
        X_train_use = X_train
        X_test_use = X_test


    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,        # Number of trees
        max_depth=20,            # Maximum depth of trees
        min_samples_split=5,     # Minimum samples to split a node
        min_samples_leaf=2,      # Minimum samples in leaf
        max_features='sqrt',     # Features to consider for best split
        random_state=RANDOM_STATE,
        n_jobs=-1,              # Use all CPU cores
        class_weight='balanced'  # Handle any class imbalance
    )
    
    print("Training Random Forest (this may take a while)...")
    rf_model.fit(X_train_use, y_train)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    print("✅ Model saved to: models/random_forest_model.pkl")
    






if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = train_random_forest()