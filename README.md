# Iris Species Predictor using Support Vector Machine (SVM)

## Overview

This project implements an Iris flower species classifier using the **Support Vector Machine (SVM)** algorithm. The application predicts the species of an Iris flower based on four user-provided flower measurements.

The model is trained on the Iris dataset and provides prediction probabilities along with graphical visualizations to help users understand the classification results.

---

## Features

- Interactive command-line interface
- Support Vector Machine (SVM) classifier
- Feature scaling using StandardScaler
- Label encoding using LabelEncoder
- Prediction confidence score
- Probability distribution chart
- SVM decision boundary visualization
- Feature comparison chart
- Prediction summary panel
- Automatic saving of visualization as an image
- Input validation

---

## Technologies Used

- Python 3
- NumPy
- Matplotlib
- Scikit-learn

---

## Project Structure

```
Iris-SVM-Predictor/
│
├── iris_svm.py
├── requirements.txt
├── README.md
└── iris_svm_prediction.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kanishk-am/Species-detection-using-SVM.git
```

Navigate to the project directory:

```bash
cd Iris-SVM-Predictor
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install numpy matplotlib scikit-learn
```

---

## Running the Project

Run the Python program:

```bash
python iris_svm.py
```

Enter the flower measurements when prompted.

Example:

```
Sepal Length : 5.1
Sepal Width  : 3.5
Petal Length : 1.4
Petal Width  : 0.2
```

The program predicts the Iris species, displays prediction probabilities, generates visualizations, and saves the output image as:

```
iris_svm_prediction.png
```

---

## Machine Learning Model

This project uses the **Support Vector Machine (SVM)** algorithm with the following configuration:

- Kernel: Radial Basis Function (RBF)
- Feature Scaling using StandardScaler
- Label Encoding using LabelEncoder
- Probability estimation enabled

The model is trained using the Iris dataset and predicts the class with the highest probability.

---

## Dataset

The project uses the Iris dataset consisting of **150 flower samples** divided into three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

Each sample contains the following features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## Output

The application displays:

- Predicted Iris species
- Prediction confidence
- Probability distribution for each class
- SVM decision boundary
- Feature comparison chart
- Prediction summary panel

The visualization is automatically saved as:

```
iris_svm_prediction.png
```

---

## Requirements

Create a `requirements.txt` file containing:

```text
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Develop a graphical user interface using Tkinter or PyQt
- Deploy as a web application using Flask or Streamlit
- Support CSV file predictions
- Display model evaluation metrics
- Add confusion matrix and classification report
- Compare SVM with other machine learning algorithms
- Deploy the model as a REST API

---

## Author

**Kanishk A M**

GitHub: https://github.com/kanishk-am
