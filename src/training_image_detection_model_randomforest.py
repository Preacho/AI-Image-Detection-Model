import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
# ---------- CONFIGURATION ----------
PREPROCESSED_DIR = "preprocessed_dataset"
IMG_WIDTH = 300
IMG_HEIGHT = 300
RANDOM_STATE = 42
NPZ_PATH = "training_data.npz"




def load_dataset_from_imageset(directory, batch_size = 64, image_size = (96,96), color_mode='grayscale'):
    data = tf.keras.utils.image_dataset_from_directory(
        directory,
        batch_size = batch_size,
        image_size = image_size, 
        color_mode = color_mode, 
        shuffle = False, 
    )
    class_names = data.class_names
    Xs, ys = [], []

    for x, y in data:
        x = x.numpy().astype(np.float32)/ 255.0
        Xs.append(x.reshape(x.shape[0], -1))
        ys.append(y.numpy())

    return np.concatenate(Xs), np.concatenate(ys), class_names 


def load_data():
    """Load train and test data from preprocessed directories"""
    
    X_train, y_train, class_names = load_dataset_from_imageset("preprocessed_dataset/train", 64, (192,192))
    X_test, y_test, _ = load_dataset_from_imageset("preprocessed_dataset/test", 64, (192,192))
    
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
        n_estimators=300,        # Number of trees
        max_depth=25,            # Maximum depth of trees
        min_samples_split=10,     # Minimum samples to split a node
        min_samples_leaf=4,      # Minimum samples in leaf
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

    y_pred = rf_model.predict(X_test) 
    y_prob = rf_model.predict_proba(X_test)[:,1]

    model_acc = accuracy_score(y_test, y_pred)
    model_prec = precision_score(y_test,y_pred)
    model_f1 = f1_score(y_test, y_pred)
    model_recall = recall_score(y_test, y_pred)
    model_aucroc = roc_auc_score(y_test,y_prob)

    print(f"Accuracy    : {model_acc:.4f}")
    print(f"Precision   : {model_prec:.4f}")
    print(f"f1   : {model_f1:.4f}")
    print(f"Recall    : {model_recall:.4f}")
    print(f"Auc Roc    : {model_aucroc:.4f}")

    






if __name__ == "__main__":
    train_random_forest()