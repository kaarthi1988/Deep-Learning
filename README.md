# Iris Classification with Deep Learning

## Project Overview

This project implements a classification model to identify iris flower species using a deep learning approach. The model is built with Keras, a high-level neural networks API running on top of TensorFlow. The Iris dataset, widely used in machine learning, serves as the basis for training and evaluating the model. This project includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset

The Iris dataset is a classic dataset in the field of machine learning and consists of 150 samples of iris flowers. Each sample includes:

- **Sepal Length** (in cm)
- **Sepal Width** (in cm)
- **Petal Length** (in cm)
- **Petal Width** (in cm)
- **Species** (the target label, with three possible values: Iris-setosa, Iris-versicolor, and Iris-virginica)

You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

## Features

- **Deep Learning Model**: Built using Keras and TensorFlow.
- **Data Preprocessing**: Includes normalization and splitting of the dataset into training and testing sets.
- **Model Training**: Training the model with the processed data.
- **Evaluation**: Model performance is assessed using accuracy metrics.

## Installation

To run this project, you'll need to install several Python packages. You can install all necessary dependencies using `pip`. Here’s how:

1. Clone the repository:

    ```bash
    git clone https://github.com/kaarthi1988/iris-classification.git
    cd iris-classification
    ```

2. Install the required packages:

    ```bash
    pip install pandas numpy tensorflow scikit-learn matplotlib
    ```

## Usage

Follow these steps to run the project:

1. **Prepare the Dataset**: Ensure that `iris.csv` is in the project directory. If not, download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) and place it in the directory.

2. **Run the Training Script**: Execute the script to train the model and generate results.

    ```bash
    python train_model.py
    ```

3. **Check Results**: The script will output the model’s performance metrics such as accuracy. It will also generate and save plots visualizing the training and validation accuracy and loss.

## Model Architecture

The model consists of:

- **Input Layer**: Accepts the feature inputs (sepal length, sepal width, petal length, petal width).
- **Hidden Layers**: Several dense layers with ReLU activation functions.
- **Output Layer**: A softmax layer for classification into three classes.

## Code Explanation

### Data Preprocessing

The data preprocessing involves:

- Loading the dataset.
- Normalizing feature values.
- Splitting the data into training and testing sets.

### Model Definition

The model is defined using Keras' Sequential API. Key components include:

- **Dense Layers**: Fully connected layers with ReLU activation.
- **Output Layer**: Softmax activation to produce class probabilities.

### Training and Evaluation

- **Compile**: Configures the model with an optimizer, loss function, and metrics.
- **Fit**: Trains the model on the training data.
- **Evaluate**: Assesses the model performance on the test data.

### Visualization

- **Accuracy and Loss Plots**: Displays the training and validation accuracy and loss over epochs.

## Results

The trained model demonstrates effective classification of iris species, achieving an accuracy of over 95% on the test set. Performance metrics and plots are saved in the `results` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions to this project are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. Ensure that your contributions are well-documented and include tests where applicable.

## Contact

For any questions or further information, you can contact me at [kaarthiprasanna@gmail.com](mailto:your.email@example.com).

## Acknowledgements

- The Iris dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).
- This project utilizes Keras and TensorFlow for building and training the deep learning model.

## Future Work

Future enhancements to this project could include:

- Experimenting with different neural network architectures.
- Implementing hyperparameter tuning.
- Exploring additional datasets for model validation.

