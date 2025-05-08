import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
import streamlit as st

def train_models(X_train, y_train, X_test, y_test):
    """
    Train Decision Tree and Random Forest models
    
    Parameters:
    X_train, y_train: Training data
    X_test, y_test: Test data
    
    Returns:
    models: Dictionary of trained models
    metrics: Dictionary of performance metrics
    feature_importances: Dictionary of feature importances
    """
    # Initialize dictionaries to store results
    models = {}
    metrics = {}
    feature_importances = {}
    
    # Train Decision Tree with hyperparameter tuning
    with st.spinner('Training Decision Tree model...'):
        param_grid_dt = {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        grid_search_dt = GridSearchCV(dt, param_grid_dt, cv=5, scoring='f1', n_jobs=-1)
        grid_search_dt.fit(X_train, y_train)
        
        # Get best model
        best_dt = grid_search_dt.best_estimator_
        models['decision_tree'] = best_dt
        
        # Make predictions
        y_pred_dt = best_dt.predict(X_test)
        y_prob_dt = best_dt.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics['decision_tree'] = {
            'accuracy': accuracy_score(y_test, y_pred_dt),
            'precision': precision_score(y_test, y_pred_dt),
            'recall': recall_score(y_test, y_pred_dt),
            'f1': f1_score(y_test, y_pred_dt),
            'confusion_matrix': confusion_matrix(y_test, y_pred_dt),
            'roc_auc': roc_auc_score(y_test, y_prob_dt),
            'y_pred': y_pred_dt,
            'y_prob': y_prob_dt
        }
        
        # Get feature importances for Decision Tree
        if hasattr(st.session_state, 'feature_names'):
            feature_importances['decision_tree'] = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': best_dt.feature_importances_
            }).sort_values('importance', ascending=False)
    
    # Train Random Forest with hyperparameter tuning
    with st.spinner('Training Random Forest model...'):
        param_grid_rf = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='f1', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        
        # Get best model
        best_rf = grid_search_rf.best_estimator_
        models['random_forest'] = best_rf
        
        # Make predictions
        y_pred_rf = best_rf.predict(X_test)
        y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics['random_forest'] = {
            'accuracy': accuracy_score(y_test, y_pred_rf),
            'precision': precision_score(y_test, y_pred_rf),
            'recall': recall_score(y_test, y_pred_rf),
            'f1': f1_score(y_test, y_pred_rf),
            'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
            'roc_auc': roc_auc_score(y_test, y_prob_rf),
            'y_pred': y_pred_rf,
            'y_prob': y_prob_rf
        }
        
        # Get feature importances for Random Forest
        if hasattr(st.session_state, 'feature_names'):
            feature_importances['random_forest'] = pd.DataFrame({
                'feature': st.session_state.feature_names,
                'importance': best_rf.feature_importances_
            }).sort_values('importance', ascending=False)
    
    return models, metrics, feature_importances

def evaluate_models(models, X_test, y_test):
    """
    Evaluate the trained models
    """
    metrics = {}
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    return metrics

def predict_single_instance(model, preprocessor, data_dict):
    """
    Make a prediction for a single client instance
    
    Parameters:
    model: Trained model
    preprocessor: Data preprocessor
    data_dict: Dictionary of client data
    
    Returns:
    prediction: Binary prediction (0 or 1)
    probability: Probability of positive class
    """
    # Convert to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Preprocess the data
    X = preprocessor.transform(df)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    return prediction, probability
