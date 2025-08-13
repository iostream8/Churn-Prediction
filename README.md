# 💳 Credit Card Customer Churn Prediction  

## 📌 Overview  
This project predicts whether a **credit card customer will churn (exit)** using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**.  
It uses the **Churn_Modelling.csv** dataset from Kaggle and covers the **full ML pipeline** from **data preprocessing** to **model evaluation**.  

---

## 📂 Dataset Information  

| Attribute       | Description |
|-----------------|-------------|
| **File**        | `Churn_Modelling.csv` |
| **Size**        | 10,000 rows × 14 columns |
| **Target**      | `Exited` (0 = retained, 1 = churned) |

### 🔑 Key Columns  

| Column            | Type        | Description |
|-------------------|-------------|-------------|
| `CreditScore`     | Numeric     | Customer’s credit score |
| `Geography`       | Categorical | Country (France, Spain, Germany) |
| `Gender`          | Categorical | Male/Female |
| `Age`             | Numeric     | Customer age |
| `Tenure`          | Numeric     | Number of years customer has stayed |
| `Balance`         | Numeric     | Account balance |
| `NumOfProducts`   | Numeric     | Bank products held |
| `HasCrCard`       | Binary      | 1 = has credit card, 0 = no card |
| `IsActiveMember`  | Binary      | 1 = active customer |
| `EstimatedSalary` | Numeric     | Estimated salary |
| `Exited`          | Binary      | Target — 1 = churned, 0 = retained |

---

## 🔄 Workflow  

| Step              | Description |
|-------------------|-------------|
| **Data Loading**  | Load dataset using `pandas.read_csv()` |
| **Cleaning**      | Drop irrelevant columns: `RowNumber`, `CustomerId`, `Surname` |
| **Encoding**      | One-hot encode `Geography`, binary encode `Gender` |
| **Scaling**       | Apply `StandardScaler` to numerical features |
| **Train/Test Split** | 80% training, 20% testing |
| **Model Building** | Build a Sequential ANN with 3 layers |
| **Training**      | Train for 100 epochs with validation |
| **Evaluation**    | Predict and compute accuracy on test data |

---

## 🧠 Model Architecture  

| Layer   | Units | Activation | Parameters |
|---------|-------|------------|------------|
| Dense_1 | 11    | ReLU       | 132        |
| Dense_2 | 11    | ReLU       | 132        |
| Output  | 1     | Sigmoid    | 12         |
| **Total** | —   | —          | **276**    |

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Metric:** Accuracy  

---

## 📊 Training Results  

| Metric    | Training Set | Validation Set | Test Set |
|-----------|-------------|----------------|----------|
| Accuracy  | ~87.85%     | ~85.50%        | **86.70%** |
| Loss      | ~0.296      | ~0.346         | ~0.347   |

---

## ▶️ How to Run  

1️⃣ **Upload** the notebook and dataset to Kaggle or run locally  
2️⃣ **Install dependencies:**  
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
