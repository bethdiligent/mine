# Configuration parameters for the emotion classifier project

# File paths
DATA_PATH = 'data/emotions.csv'
MODEL_DIR = 'models/'

# Preprocessing parameters
VARIANCE_THRESHOLD = 0.01

# Model hyperparameters
DECISION_TREE_PARAMS = {
    'criterion': 'gini',
    'max_depth': None,
    'min_samples_split': 2,
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
}

# Other parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2