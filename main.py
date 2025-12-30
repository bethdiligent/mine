"""
Main entry point of the emotion classification project.

This script handles the following tasks:
1. Load the dataset.
2. Preprocess the data.
3. Train machine learning models.
4. Evaluate the models.
5. Save the trained models for future use.
"""

import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from preprocessing import DataPreprocessor
from models import train_decision_tree, train_svm, evaluate_model
import joblib

def main():
    # 1. Load data
    print("Loading data...")
    df = pd.read_csv('data/emotions.csv')
    
    # 2. Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor(variance_threshold=0.01)
    
    # Split data before preprocessing
    from sklearn.model_selection import train_test_split
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocess training and test data
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # 3. Save preprocessor for future use
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    # 4. Train models
    print("\nTraining Decision Tree...")
    dt_model = train_decision_tree(X_train_processed, y_train_processed)
    
    print("\nTraining SVM...")
    svm_model = train_svm(X_train_processed, y_train_processed)
    
    # 5. Evaluate models
    print("\nEvaluating models...")
    dt_results = evaluate_model(dt_model, X_test_processed, y_test_processed, "Decision Tree")
    svm_results = evaluate_model(svm_model, X_test_processed, y_test_processed, "SVM")
    
    # 6. Save models
    joblib.dump(dt_model, 'models/decision_tree.pkl')
    joblib.dump(svm_model, 'models/svm.pkl')
    
    print("\n✅ All models saved successfully!")

if __name__ == "__main__":
    main()
