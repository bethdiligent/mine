# Emotion Classifier

This project is an Emotion Classifier that utilizes machine learning techniques to classify emotions based on a dataset. The project is structured to facilitate exploratory data analysis, preprocessing, model training, and evaluation.

## Project Structure

```
emotion-classifier
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_experiments.ipynb
├── src
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── utils.py
├── config
│   └── params.py
├── data
│   └── emotions.csv
├── models
├── tests
│   └── test_preprocessing.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd emotion-classifier
pip install -r requirements.txt
```

## Usage

1. **Data Exploration**: Use the Jupyter Notebook `notebooks/01_data_exploration.ipynb` to perform exploratory data analysis on the dataset.
2. **Data Preprocessing**: The `notebooks/02_preprocessing.ipynb` notebook is used for preprocessing the data, including handling missing values and scaling features.
3. **Model Experiments**: Experiment with different machine learning models in `notebooks/03_model_experiments.ipynb`.
4. **Run the Main Script**: Execute `main.py` to load the data, preprocess it, train the models, and evaluate their performance.

## Testing

Unit tests for the preprocessing functions can be found in `tests/test_preprocessing.py`. Run the tests to ensure that the preprocessing functions work as expected.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.