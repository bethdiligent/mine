import pytest
import pandas as pd
from src.preprocessing import DataPreprocessor

def test_handle_missing_values():
    # Create a sample DataFrame with missing values
    data = {
        'feature1': [1, 2, None, 4],
        'feature2': ['a', 'b', 'b', None],
        'label': ['happy', 'sad', 'happy', 'sad']
    }
    df = pd.DataFrame(data)

    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.fit_transform(df.drop('label', axis=1), df['label'])

    # Check if missing values are handled
    assert df_cleaned.isnull().sum().sum() == 0

def test_feature_selection():
    # Create a sample DataFrame with low variance feature
    data = {
        'feature1': [1, 1, 1, 1],
        'feature2': [1, 2, 3, 4],
        'label': ['happy', 'sad', 'happy', 'sad']
    }
    df = pd.DataFrame(data)

    preprocessor = DataPreprocessor(variance_threshold=0.01)
    X_processed, _ = preprocessor.fit_transform(df.drop('label', axis=1), df['label'])

    # Check if low variance feature is removed
    assert X_processed.shape[1] == 1  # Only feature2 should remain

def test_scaling():
    # Create a sample DataFrame
    data = {
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40],
        'label': ['happy', 'sad', 'happy', 'sad']
    }
    df = pd.DataFrame(data)

    preprocessor = DataPreprocessor()
    X_processed, _ = preprocessor.fit_transform(df.drop('label', axis=1), df['label'])

    # Check if the mean is approximately 0 and standard deviation is approximately 1
    assert abs(X_processed.mean()) < 1e-5
    assert abs(X_processed.std() - 1) < 1e-5

def test_label_encoding():
    # Create a sample DataFrame
    data = {
        'feature1': [1, 2, 3, 4],
        'label': ['happy', 'sad', 'happy', 'sad']
    }
    df = pd.DataFrame(data)

    preprocessor = DataPreprocessor()
    _, y_encoded = preprocessor.fit_transform(df.drop('label', axis=1), df['label'])

    # Check if labels are encoded correctly
    assert set(y_encoded) == {0, 1}  # Assuming 'happy' is 0 and 'sad' is 1