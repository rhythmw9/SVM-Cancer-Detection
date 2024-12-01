# SVM Classifier for Breast Cancer Detection

This project demonstrates the implementation of a Support Vector Machine (SVM) classifier to detect benign and malignant tumors based on the **Wisconsin Breast Cancer Dataset**. The project focuses on data preprocessing, training an SVM model with various hyperparameter values, and evaluating its performance.

---

## Features

- **Preprocessing:**
  - Cleans the dataset (handles missing values).
  - Standardizes features for SVM compatibility.
  - Splits data into training and testing sets.
  
- **Training:**
  - Implements SVM optimization using CVXPY.
  - Tests the model with multiple values of the regularization hyperparameter (\(\gamma\)).
  
- **Evaluation:**
  - Computes training and testing errors for each value of \(\gamma\).
  - Identifies the best \(\gamma\) value based on the model's performance.
  - Visualizes training and testing errors as a function of \(\gamma\).

---

## Directory Structure

SVM-Classifier/              # Main project folder
├── data/                    # Folder for storing datasets
│   └── breast-cancer-wisconsin.data  # dataset file
├── src/                     # Folder for source code
│   ├── data_preprocessing.py         # Script for preprocessing the dataset
│   ├── SVM_model.py                  # Script for SVM training logic
│   ├── evaluation.py                 # Script for evaluation and plotting
├── main.py                  # entry point
├── README.md                # Documentation
├── requirements.txt         # File listing dependencies
└── .gitignore               # Specifies which files/folders Git should ignore



---

## How to Run

### Prerequisites
- Python 3.7+
- Recommended: A virtual environment to manage dependencies.

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/SVM-Classifier.git
   cd SVM-Classifier

## Results

### Best Gamma Value
- **Best Gamma (\(\gamma\)):** `0.1`
- **Train Error:** `3%`
- **Test Error:** `3%`


## Requirements

The following libraries are required to run the project:

- `cvxpy==1.2.3`
- `matplotlib==3.6.2`
- `numpy==1.23.5`
- `pandas==1.5.2`
- `scikit-learn==1.1.3`

To install all dependencies, run:
```bash
pip install -r requirements.txt


---

### **Dataset Information**

```markdown
## Dataset

The project uses the **Wisconsin Breast Cancer Dataset**, included in the `data/` folder for reproducibility. 

### Dataset Overview
The dataset contains features derived from digitized images of fine needle aspirates (FNAs) of breast masses. It is widely used for binary classification tasks in machine learning.

| Feature        | Description                  |
|----------------|------------------------------|
| Clump Thickness | Thickness of clump          |
| Uniformity      | Uniformity of cell size/shape |
| Marginal Adhesion | Adhesion of epithelial cells |
| ...            | (other features)            |

### Target Values
- **2:** Benign (non-cancerous)
- **4:** Malignant (cancerous)

For more information, see the dataset source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)).

## Future Improvements

This project can be extended in the following ways:

1. **Dataset Enhancements**:
   - Use a larger or more diverse dataset to improve generalization.
   - Experiment with other datasets for multi-class classification.

2. **Model Optimization**:
   - Add cross-validation for hyperparameter tuning.
   - Use non-linear kernels or other advanced SVM formulations.

3. **Evaluation Metrics**:
   - Extend the evaluation to include precision, recall, F1-score, and AUC.

4. **Visualization**:
   - Add more visualizations, such as decision boundaries or feature importance.

## Author and License

### Author
- **Rhythm Winicour-Freeman**  
- GitHub: [Your GitHub Profile](https://github.com/rhythmw9)  
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/rhythm-winicour-freeman-975b74289/)

### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
