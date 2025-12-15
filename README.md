# 🌲 Multi-Class Classification with SHAP Explainability

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-V3.1.1-green?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange?style=for-the-badge&logo=shap&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)


## 🎯 Project Objective
The goal of this project is to build a robust multi-class classifier to predict forest cover types and interpret its predictions using SHAP (SHapley Additive exPlanations). This ensures not only high predictive performance but also transparency and trust in the model's decision-making process.

## ❓ Problem Statement
1.  **Build a Multi-Class Classifier**: Develop a machine learning model to categorize data into multiple classes.
2.  **Explain Predictions**: Use SHAP or LIME to explain individual predictions.
3.  **Feature Preprocessing**: Handle multicollinearity and apply scaling where necessary.
4.  **Training & Evaluation**: Train the model using cross-validation and evaluate performance using:
    *   Accuracy
    *   Confusion Matrix
    *   Classification Report
5.  **Interpretability**: Visualize and explain key feature contributions for specific predictions.

## 📂 Dataset
**Forest Cover Type Dataset** (`covtype.data.gz`)
This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado. All observations are cartographic variables (no remote sensing) from 30 meter x 30 meter sections of forest.

## 🏗️ Project Structure
*   `Forest_CoverType_Multiclass_SHAP.ipynb`: The main Jupyter Notebook containing the end-to-end workflow:
    *   Data Loading & Exploration
    *   Preprocessing (Scaling, etc.)
    *   Model Training (XGBoost Classifier)
    *   Evaluation (Confusion Matrix, Classification Report)
    *   Explainability (SHAP KernelExplainer)
*   `SHAP_Explanation_and_Fix.md`: A detailed report on the challenges faced with SHAP and XGBoost versions, and the implemented solution using `KernelExplainer`.
*   `covtype.data.gz`: Reduced version of the dataset used for this assignment.

## ⚙️ Methodology

### 1. 🧹 Preprocessing
*   **Data Loading**: Reading the compressed data file.
*   **Scaling**: Standardizing features to ensure uniform contribution to the model (though tree-based models are robust to this, it's good practice).
*   **Multicollinearity Check**: Analyzing feature correlations to ensure model stability.

### 2. 🚀 Model Training
*   **Algorithm**: **XGBoost Classifier** (Extreme Gradient Boosting) was chosen for its efficiency and high performance on structured data.
*   **Cross-Validation**: Used to ensure the model generalizes well to unseen data.

### 3. 📊 Evaluation
The model is evaluated on a hold-out test set using:
*   **Accuracy Score**: Overall correctness of the model.
*   **Classification Report**: Precision, Recall, and F1-Score for each class.
*   **Confusion Matrix**: Visualizing misclassifications across different cover types.

### 4. 🧠 Explainability (SHAP)
Due to version incompatibilities between recent XGBoost (v3.1.1) and SHAP (v0.48.0), we implemented `shap.KernelExplainer` as a model-agnostic solution.
*   **KernelExplainer**: Approximates SHAP values by sampling coalitions of features.
*   **Wrapper Function**: A custom wrapper (`predict_wrapper`) ensures seamless integration with the XGBoost model.
*   **Visualizations**:
    *   **Force Plot**: Visualizes the contribution of each feature to a specific prediction (pushing the score higher or lower).
    *   **Summary Plot**: Shows the global importance of features and their impact on different classes.

## 🏃‍♂️ How to Run
1.  Ensure you have the required libraries installed:
    ```bash
    pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn
    ```
2.  Open `Forest_CoverType_Multiclass_SHAP.ipynb` in Jupyter Notebook or VS Code.
3.  Run all cells to execute the pipeline. 
    *   *Note*: The SHAP calculations using `KernelExplainer` may take a few moments due to the computational complexity of the method.

## 👏 Acknowledgments
*   Dataset provided for machine learning assignment purposes.
*   SHAP library by Scott Lundberg for model interpretability.
