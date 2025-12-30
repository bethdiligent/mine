def create_feature_engineering_features(df):
    """Create new features for the dataset."""
    
    # Example: Create a feature for the length of text in a column named 'text'
    if 'text' in df.columns:
        df['text_length'] = df['text'].apply(len)
    
    # Example: Create a feature for the number of exclamation marks in the 'text' column
    if 'text' in df.columns:
        df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    
    # Example: Create a feature for the number of question marks in the 'text' column
    if 'text' in df.columns:
        df['question_count'] = df['text'].apply(lambda x: x.count('?'))
    
    return df

def transform_features(df):
    """Transform existing features in the dataset."""
    
    # Example: Normalize a numerical feature named 'numerical_feature'
    if 'numerical_feature' in df.columns:
        df['numerical_feature'] = (df['numerical_feature'] - df['numerical_feature'].mean()) / df['numerical_feature'].std()
    
    return df