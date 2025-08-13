Credit Card Customer Churn Prediction
📌 Overview
This project predicts whether a credit card customer will churn (exit) using a Neural Network (ANN) built with TensorFlow/Keras.
The dataset used is Churn_Modelling.csv from the Kaggle dataset: Credit Card Customer Churn Prediction.
The notebook walks through:

Data loading & preprocessing

Feature encoding & scaling

Building and training an ANN model

Model evaluation

Accuracy and loss visualization

📂 Dataset
File: Churn_Modelling.csv
Size: 10,000 rows × 14 columns

Key columns:
CreditScore: Customer credit score

Geography: Country (France, Spain, Germany)

Gender: Male/Female

Age

Tenure: Years of being a customer

Balance

NumOfProducts: Number of bank products

HasCrCard: 1 = has a credit card

IsActiveMember: 1 = active

EstimatedSalary

Exited: Target variable (1 = churned, 0 = retained)

🔄 Workflow
1️⃣ Data Loading
Loads the CSV from the Kaggle input directory:

python
df = pd.read_csv('/kaggle/input/credit-card-customer-churn-prediction/Churn_Modelling.csv')
2️⃣ Preprocessing
Drop irrelevant columns (RowNumber, CustomerId, Surname)

Encode categorical variables:

Geography → One-Hot Encoding

Gender → Binary Encoding

Combine into a clean numerical feature set

Split into x_train, x_test, y_train, y_test

Apply StandardScaler

3️⃣ Model Architecture
Sequential ANN model:

text
Input Layer → Dense(11, relu)
               ↓
            Dense(11, relu)
               ↓
            Dense(1, sigmoid)
Loss: binary_crossentropy

Optimizer: Adam

Metric: accuracy

4️⃣ Training
Epochs: 100

Batch size: (default by Keras)

Validation split: 0.2

History recorded for accuracy/loss plotting

5️⃣ Evaluation
Predict probabilities → Convert to 0 or 1 (>0.5)

Accuracy score on test set ≈ 86.7%

Model training/validation curves stored in history.history

📊 Example Accuracy/Loss
text
Final Training Accuracy: ~87.8%
Final Validation Accuracy: ~85.5%
Test Accuracy: 0.867
▶️ How to Run
Upload to Kaggle or run locally with Jupyter Notebook.

Install requirements (if not using Kaggle):

bash
pip install numpy pandas scikit-learn tensorflow matplotlib
Place the Churn_Modelling.csv file in the correct directory.

Run all cells sequentially to:

Preprocess data

Train the model

Evaluate performance

📈 Possible Improvements
Tune hyperparameters (layers, neurons, learning rate)

Use dropout or regularization to reduce overfitting

Try advanced models (XGBoost, LightGBM)

Balance the dataset (SMOTE/undersampling)

🛠 Dependencies
Python 3.x

numpy

pandas

scikit-learn

tensorflow / keras

matplotlib

