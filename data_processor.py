import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
import io
import requests
import zipfile

def load_data():
    """
    Load the bank marketing dataset from UCI Repository
    """
    # URL for the Bank Marketing dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    
    try:
        # Download the zip file
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Read the zip file
            zip_file = io.BytesIO(response.content)
            
            # Extract the CSV file from the zip
            with zipfile.ZipFile(zip_file) as z:
                # Get the name of the CSV file (assuming there's only one or we know the name)
                csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
                with z.open(csv_filename) as f:
                    # Read the CSV file
                    df = pd.read_csv(f, delimiter=';')
            
            return df
        else:
            raise Exception(f"Failed to download data: Status code {response.status_code}")
    except Exception as e:
        # If there's an error, use a direct CSV link as backup
        backup_url = "https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv"
        st.warning(f"Error accessing UCI repository: {str(e)}. Using backup source.")
        
        response = requests.get(backup_url)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text), delimiter=';')
            return df
        else:
            raise Exception("Failed to load data from all sources")

def preprocess_data(df):
    """
    Preprocess the bank marketing dataset:
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    - Split into train and test sets
    
    Returns:
    - X_train, X_test, y_train, y_test: Train and test splits
    - df_processed: Processed dataframe for EDA
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert target variable 'y' to binary (1 for 'yes', 0 for 'no')
    df_processed['y'] = df_processed['y'].map({'yes': 1, 'no': 0})
    
    # Define features and target
    X = df_processed.drop('y', axis=1)
    y = df_processed['y']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform the test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    ohe_feature_names = []
    
    # Add numerical feature names
    ohe_feature_names.extend(numerical_cols)
    
    # Add categorical feature names with their one-hot encoded versions
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    for i, col in enumerate(categorical_cols):
        categories = ohe.categories_[i]
        for category in categories:
            ohe_feature_names.append(f"{col}_{category}")
    
    # Store the preprocessor and feature names in session state for later use
    st.session_state.preprocessor = preprocessor
    st.session_state.feature_names = ohe_feature_names
    
    return X_train_processed, X_test_processed, y_train, y_test, df_processed

def get_feature_names():
    """
    Get feature names from session state
    """
    if 'feature_names' not in st.session_state:
        raise ValueError("Feature names not available. Please run preprocessing first.")
    
    return st.session_state.feature_names
