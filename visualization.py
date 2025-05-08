import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
import streamlit as st

def plot_feature_importance(importance_df, model_name, n_features=15):
    """
    Plot feature importance
    
    Parameters:
    importance_df: DataFrame with feature importances
    model_name: Name of the model
    n_features: Number of top features to show
    """
    if importance_df is None:
        return None
    
    # Get top n features
    top_features = importance_df.head(n_features)
    
    # Create a horizontal bar chart
    fig = px.bar(
        top_features,
        y='feature',
        x='importance',
        orientation='h',
        title=f'Top {n_features} Feature Importances - {model_name}',
        labels={'importance': 'Importance', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='viridis'
    )
    
    # Update layout
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500,
        width=700
    )
    
    return fig

def plot_confusion_matrix(cm, model_name):
    """
    Plot confusion matrix
    
    Parameters:
    cm: Confusion matrix
    model_name: Name of the model
    """
    # Calculate derived metrics
    tn, fp, fn, tp = cm.ravel()
    
    # Create a heatmap for the confusion matrix
    fig = make_subplots(rows=1, cols=1)
    
    # Add heatmap trace
    heatmap = go.Heatmap(
        z=cm,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        colorscale='Blues',
        showscale=True,
        text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']],
        texttemplate="%{text}",
        textfont={"size": 15}
    )
    
    fig.add_trace(heatmap)
    
    # Update layout
    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        height=500,
        width=500,
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    
    return fig

def plot_roc_curve(metrics_dict):
    """
    Plot ROC curve for all models
    
    Parameters:
    metrics_dict: Dictionary of model metrics
    
    Returns:
    fig: Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Get ROC curve for each model
    for model_name, metrics in metrics_dict.items():
        y_true = st.session_state.y_test if 'y_test' in st.session_state else None
        y_score = metrics.get('y_prob', None)
        
        if y_true is not None and y_score is not None:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = metrics.get('roc_auc', auc(fpr, tpr))
            
            # Add ROC curve to plot
            display_name = 'Decision Tree' if model_name == 'decision_tree' else 'Random Forest'
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=f'{display_name} (AUC = {roc_auc:.4f})',
                    mode='lines',
                    line=dict(width=2)
                )
            )
    
    # Add diagonal line (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random Classifier (AUC = 0.5)',
            mode='lines',
            line=dict(dash='dash', color='gray')
        )
    )
    
    # Update layout
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
        width=700,
        height=500
    )
    
    # Update axes
    fig.update_xaxes(constrain='domain')
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def plot_distribution(df, column, hue=None):
    """
    Plot distribution of a column
    
    Parameters:
    df: DataFrame
    column: Column to plot
    hue: Column to use for color
    """
    if column not in df.columns:
        st.error(f"Column '{column}' not found in dataframe")
        return None
    
    # Check if column is numerical or categorical
    if df[column].dtype in ['int64', 'float64']:
        # Numerical column - histogram
        if hue is None:
            fig = px.histogram(
                df,
                x=column,
                title=f'Distribution of {column}',
                color_discrete_sequence=['#3366CC'],
                marginal='box'
            )
        else:
            fig = px.histogram(
                df,
                x=column,
                color=hue,
                title=f'Distribution of {column} by {hue}',
                barmode='overlay',
                marginal='box',
                opacity=0.7
            )
    else:
        # Categorical column - bar chart
        if hue is None:
            value_counts = df[column].value_counts().reset_index()
            value_counts.columns = [column, 'count']
            
            fig = px.bar(
                value_counts,
                x=column,
                y='count',
                title=f'Distribution of {column}',
                color_discrete_sequence=['#3366CC']
            )
        else:
            cross_tab = pd.crosstab(df[column], df[hue])
            cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
            
            fig = px.bar(
                cross_tab_pct,
                title=f'Distribution of {column} by {hue}',
                barmode='group'
            )
    
    # Update layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Count' if hue is None else 'Percentage (%)',
        height=500,
        width=700
    )
    
    return fig

def plot_correlation_matrix(df):
    """
    Plot correlation matrix
    
    Parameters:
    df: DataFrame
    target: Target variable to highlight
    """
    # Get numerical columns
    num_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation matrix
    corr_matrix = num_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        text_auto='.2f'
    )
    
    # Update layout
    fig.update_layout(
        title='Correlation Matrix',
        height=800,
        width=800
    )
    
    return fig

def plot_feature_target_relationship(df, feature, target='y'):
    """
    Plot relationship between a feature and the target
    
    Parameters:
    df: DataFrame
    feature: Feature column
    target: Target column
    """
    if feature not in df.columns or target not in df.columns:
        st.error(f"One or both columns not found in dataframe: {feature}, {target}")
        return None
    
    # Check if feature is categorical or numerical
    if df[feature].dtype in ['int64', 'float64']:
        # Numerical feature - box plot
        fig = px.box(
            df,
            x=target,
            y=feature,
            title=f'Relationship between {feature} and {target}',
            color=target,
            notched=True
        )
        
        # Add jittered points
        fig.add_trace(
            go.Scatter(
                x=df[target] + np.random.normal(0, 0.05, size=len(df)),
                y=df[feature],
                mode='markers',
                marker=dict(
                    color='rgba(0, 0, 0, 0.2)',
                    size=3
                ),
                showlegend=False
            )
        )
    else:
        # Categorical feature - stacked bar chart
        cross_tab = pd.crosstab(df[feature], df[target])
        cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100
        
        fig = px.bar(
            cross_tab_pct,
            title=f'Subscription Rate by {feature}',
            barmode='stack'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=feature,
            yaxis_title='Percentage (%)'
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=700
    )
    
    return fig

def plot_model_metrics_comparison(metrics_dict):
    """
    Plot a comparison of model metrics
    
    Parameters:
    metrics_dict: Dictionary of model metrics
    
    Returns:
    fig: Plotly figure
    """
    # Extract metrics for each model
    models = list(metrics_dict.keys())
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metrics_display = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    # Create data for plotting
    data = []
    for model in models:
        model_metrics = metrics_dict[model]
        model_values = [model_metrics[metric] for metric in metrics_list]
        
        # Add trace for each model
        display_name = 'Decision Tree' if model == 'decision_tree' else 'Random Forest'
        data.append(
            go.Bar(
                name=display_name,
                x=metrics_display,
                y=model_values
            )
        )
    
    # Create figure
    fig = go.Figure(data=data)
    
    # Update layout
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Metric',
        yaxis_title='Value',
        barmode='group',
        yaxis=dict(range=[0, 1]),
        width=700,
        height=500
    )
    
    return fig
