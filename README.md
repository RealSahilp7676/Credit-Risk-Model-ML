# Credit Risk Modeling Project

## Overview
This project implements a credit risk modeling solution using logistic regression to predict loan default risk. It utilizes customer, loan, and credit bureau data to build a predictive model, incorporating data preprocessing, feature engineering, model training, and evaluation.

## Project Structure
- **Notebook**: `credit_risk_model_codebasics.ipynb` - Main Jupyter Notebook containing the code for data loading, preprocessing, modeling, and evaluation.
- **Dataset**: 
  - `dataset/customers.csv`: Contains customer demographic information (e.g., age, gender, income).
  - `dataset/loans.csv`: Contains loan details (e.g., loan amount, tenure, default status).
  - `dataset/bureau_data.csv`: Contains credit bureau data (e.g., open accounts, credit utilization).
- **Artifacts**: 
  - `artifacts/model_data.joblib`: Saved model file containing the trained logistic regression model, feature names, scaler, and columns to scale.

## Prerequisites
To run this project, ensure you have the following Python libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

You can install them using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## Data Description
The dataset consists of three CSV files:
1. **customers.csv**: Includes customer details such as:
   - `cust_id`: Unique customer identifier
   - `age`, `gender`, `marital_status`, `employment_status`, `income`, etc.
2. **loans.csv**: Includes loan details such as:
   - `loan_id`, `cust_id`, `loan_purpose`, `loan_type`, `sanction_amount`, `default`, etc.
3. **bureau_data.csv**: Includes credit bureau data such as:
   - `cust_id`, `number_of_open_accounts`, `credit_utilization_ratio`, `delinquent_months`, etc.

Each dataset contains 50,000 records, which are merged on `cust_id` for analysis.

## Methodology
1. **Data Loading and Merging**:
   - Load the three datasets using pandas.
   - Merge `customers.csv` and `loans.csv` on `cust_id`, then merge with `bureau_data.csv` to create a unified dataset.
2. **Feature Engineering**:
   - Create new features like `loan_to_income` and `avg_dpd_per_delinquency`.
   - Encode categorical variables (e.g., `residence_type`, `loan_purpose`, `loan_type`) using one-hot encoding.
   - Scale numerical features using a scaler (saved in `model_data.joblib`).
3. **Model Training**:
   - Use logistic regression to predict the `default` column (True/False).
   - Split data into training and testing sets using `train_test_split`.
   - Train the model and evaluate feature importance based on model coefficients.
4. **Model Saving**:
   - Save the trained model, feature names, scaler, and columns to scale in `artifacts/model_data.joblib`.

## Usage
1. Clone the repository or download the project files.
2. Ensure the dataset files are in the `dataset/` directory.
3. Open and run the `credit_risk_model_codebasics.ipynb` notebook in a Jupyter environment.
4. The notebook will:
   - Load and preprocess the data.
   - Train the logistic regression model.
   - Display feature importance using a bar plot.
   - Save the model to `artifacts/model_data.joblib`.

## Results
- The logistic regression model is trained to predict loan defaults.
- Feature importance is visualized to show which features (e.g., `credit_utilization_ratio`, `loan_to_income`) most influence the prediction.
- The model and preprocessing components are saved for future use.

## Future Improvements
- Experiment with other algorithms (e.g., Random Forest, XGBoost) for better performance.
- Perform hyperparameter tuning to optimize the logistic regression model.
- Add cross-validation to ensure robust model evaluation.
- Include additional feature engineering to capture more complex patterns.

## License
This project is licensed under the MIT License.
