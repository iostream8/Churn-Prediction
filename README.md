Credit Card Customer Churn Prediction
📌 Overview
This project predicts whether a credit card customer will churn (exit) using a Neural Network (ANN) built with TensorFlow/Keras.
It uses the Churn_Modelling.csv dataset from Kaggle and covers the full pipeline from data preprocessing to model evaluation.

📂 Dataset Information
Attribute	Description
File	Churn_Modelling.csv
Size	10,000 rows × 14 columns
Target Variable	Exited (0 = retained, 1 = churned)
Key Columns
Column	Type	Description
CreditScore	Numeric	Customer’s credit score
Geography	Categorical	Country (France, Spain, Germany)
Gender	Categorical	Male/Female
Age	Numeric	Customer age
Tenure	Numeric	Number of years customer has stayed
Balance	Numeric	Account balance
NumOfProducts	Numeric	Bank products held
HasCrCard	Binary	1 = has a credit card, 0 = no card
IsActiveMember	Binary	1 = active customer
EstimatedSalary	Numeric	Estimated salary
Exited	Binary	Target — 1 = churned, 0 = retained
🔄 Workflow
Step	Description
1. Data Loading	Load dataset using pandas.read_csv()
2. Cleaning	Drop irrelevant columns: RowNumber, CustomerId, Surname
3. Encoding	One-hot encode Geography, binary encode Gender
4. Scaling	Apply StandardScaler to numerical features
5. Train/Test Split	80% training, 20% testing
6. Model Building	Build a Sequential ANN with 3 layers
7. Training	Train for 100 epochs with validation
8. Evaluation	Predict and compute accuracy on test data
🧠 Model Architecture
Layer	Units	Activation	Parameters
Dense_1	11	ReLU	132
Dense_2	11	ReLU	132
Output	1	Sigmoid	12
Total Parameters: 276
Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Accuracy

📊 Training Results
Metric	Training Set	Validation Set	Test Set
Accuracy	~87.85%	~85.50%	86.70%
Loss	~0.296	~0.346	~0.347
▶️ How to Run
Step	Command / Action
1	Upload notebook and dataset to Kaggle or run locally
2	Install dependencies: pip install numpy pandas scikit-learn tensorflow matplotlib
3	Ensure Churn_Modelling.csv is in the same directory
4	Run cells sequentially to preprocess, train, and evaluate
🚀 Possible Improvements
Area	Suggestion
Hyperparameters	Tune epochs, batch size, learning rate
Architecture	Add dropout layers for regularization
Alternative Models	Try XGBoost, LightGBM
Data Balancing	Use SMOTE or undersampling to address class imbalance
🛠 Dependencies
Package	Version (Recommended)
Python	3.x
numpy	>=1.19
pandas	>=1.1
scikit-learn	>=0.24
tensorflow/keras	>=2.6
matplotlib	>=3.3
