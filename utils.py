def load_data(file_path):
    """Load data from a CSV file."""
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(df, file_path):
    """Save DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)

def plot_distribution(data, column, title='Distribution', xlabel='Values', ylabel='Frequency'):
    """Plot the distribution of a specified column in the DataFrame."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def display_missing_values(df):
    """Display missing values in the DataFrame."""
    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100
    return pd.DataFrame({'Missing Count': missing, 'Missing Percentage': missing_percentage}).loc[missing > 0]