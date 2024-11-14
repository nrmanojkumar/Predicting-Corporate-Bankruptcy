import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('bank_new.csv')

# Separate features and target
y = df[['Bankrupt?']]
X = df.drop('Bankrupt?', axis=1)

# Scaling the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Identify and remove outliers using IsolationForest
iso_forest = IsolationForest(contamination=0.01)
outliers = iso_forest.fit_predict(X_scaled)
mask = outliers != -1
X_no_outliers = X_scaled[mask]
y_no_outliers = y[mask]

# Split the data without outliers into training and test sets
X_train_no_outliers, X_test_no_outliers, y_train_no_outliers, y_test_no_outliers = train_test_split(X_no_outliers,
                                                                                                    y_no_outliers,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)
# Train final model: Decision Trees
dtc = DecisionTreeClassifier()

# Include Hyperparameters
param_grids = {'max_depth': [None, 10, 20, 30, 40, 50], 'criterion': ['gini', 'entropy']}


# Function to calculate and store metrics
def calculate_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1


# Train the model
grid_search = GridSearchCV(dtc, param_grids, cv=5, scoring='accuracy')
grid_search.fit(X_train_no_outliers, y_train_no_outliers)
best_model = grid_search.best_estimator_
y_pred_no_outliers = best_model.predict(X_test_no_outliers)
no_outliers_scores = calculate_metrics(y_test_no_outliers, y_pred_no_outliers)

# Model Deployment

import streamlit as st

st.title('Bankruptcy Predictor')

# Input for company name
company_name = st.text_input('Company Name', key='company_name')

# Initialize session state for input fields
if 'input_data' not in st.session_state:
    st.session_state.input_data = {feature: 0.0 for feature in X.columns}

# Input fields for each feature
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f'{feature}', value=st.session_state.input_data[feature], key=feature)

# Update session state with current inputs
st.session_state.input_data.update(input_data)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize the input features
input_scaled = scaler.transform(input_df)

# Predict button
if st.button('Predict Bankruptcy'):
    # Make predictions
    prediction = best_model.predict(input_scaled)
    result = 'is likely to get BANKRUPT!' if prediction[0] == 1 else 'is NOT likely to get BANKRUPT!'
    st.write(f'{company_name} {result}')

# Refresh button to clear inputs
if st.button('Refresh'):
    for feature in X.columns:
        st.session_state.input_data[feature] = 0.0
    st.experimental_rerun()
