import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import load_data, preprocess_data, get_feature_names
from model_trainer import train_models, evaluate_models, predict_single_instance
from visualization import (
    plot_feature_importance, 
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_distribution, 
    plot_correlation_matrix,
    plot_model_metrics_comparison
)
import time

# Set page config
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Bank Marketing Prediction Dashboard")
st.markdown("""
This application uses Decision Tree and Random Forest models to predict the likelihood of clients subscribing to a bank term deposit.
It also provides insights into the key factors that influence these decisions to help optimize marketing campaigns.
""")

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'feature_importances' not in st.session_state:
    st.session_state.feature_importances = {}

# Main page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Loading", "Exploratory Data Analysis", "Model Training", "Performance Evaluation", "Feature Importance"])

# Data Loading section
if page == "Data Loading":
    st.header("Data Loading and Processing")
    
    # Load data
    if not st.session_state.data_loaded:
        if st.button("Load Bank Marketing Data"):
            data_load_state = st.text('Loading data...')
            try:
                # Load data
                df = load_data()
                
                # Store in session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                data_load_state.text('Data loaded successfully!')
                st.success(f"✅ Data loaded with {df.shape[0]} records and {df.shape[1]} features!")
            except Exception as e:
                data_load_state.error(f'Error loading data: {str(e)}')
    else:
        st.success('✅ Data already loaded!')
    
    # Preprocess data
    if st.session_state.data_loaded and ('X_train' not in st.session_state or st.session_state.X_train is None):
        if st.button("Preprocess Data"):
            preprocess_state = st.text('Preprocessing data...')
            try:
                # Preprocess data
                X_train, X_test, y_train, y_test, df_processed = preprocess_data(st.session_state.df)
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.df_processed = df_processed
                
                preprocess_state.text('Data preprocessed successfully!')
                st.success(f"✅ Data preprocessed with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples!")
            except Exception as e:
                preprocess_state.error(f'Error preprocessing data: {str(e)}')
    elif st.session_state.data_loaded and 'X_train' in st.session_state and st.session_state.X_train is not None:
        st.success('✅ Data already preprocessed!')
    
    # Display data overview
    if st.session_state.data_loaded:
        st.subheader("Data Overview")
        
        # Show basic statistics
        if st.checkbox("Show dataset shape and types"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**DataFrame Information:**")
                st.write(f"- Number of rows: {st.session_state.df.shape[0]}")
                st.write(f"- Number of columns: {st.session_state.df.shape[1]}")
            
            with col2:
                st.write("**Target variable distribution:**")
                target_counts = st.session_state.df['y'].value_counts()
                st.write(f"- 'yes': {target_counts.get('yes', 0)} ({target_counts.get('yes', 0)/len(st.session_state.df):.2%})")
                st.write(f"- 'no': {target_counts.get('no', 0)} ({target_counts.get('no', 0)/len(st.session_state.df):.2%})")
        
        # Show sample data
        if st.checkbox("Show sample data"):
            st.write("**Sample data (first 10 rows):**")
            st.write(st.session_state.df.head(10))
        
        # Show column info
        if st.checkbox("Show column information"):
            col_info = pd.DataFrame({
                'Column': st.session_state.df.columns,
                'Type': st.session_state.df.dtypes,
                'Non-Null Count': st.session_state.df.count(),
                'Unique Values': [st.session_state.df[col].nunique() for col in st.session_state.df.columns]
            })
            st.write(col_info)

# Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the data first in the Data Loading section.")
    else:
        # Access the data
        df = st.session_state.df
        
        # EDA Sections
        eda_section = st.selectbox(
            "Select Analysis Type",
            ["Variable Distributions", "Correlation Analysis", "Target Analysis"]
        )
        
        if eda_section == "Variable Distributions":
            st.subheader("Variable Distributions")
            
            # Select variable type
            var_type = st.radio("Variable type", ["Numerical", "Categorical"])
            
            if var_type == "Numerical":
                # Get numerical columns
                num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                
                # Select column
                selected_col = st.selectbox("Select numerical column", num_cols)
                
                # Create distribution plot
                fig = plot_distribution(df, selected_col)
                st.plotly_chart(fig)
                
                # Show distribution by target
                st.subheader(f"{selected_col} by Subscription Status")
                fig = plot_distribution(df, selected_col, "y")
                st.plotly_chart(fig)
                
                # Show summary statistics
                st.subheader(f"Summary Statistics for {selected_col}")
                st.write(df[selected_col].describe())
                
            else:  # Categorical
                # Get categorical columns
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                
                # Select column
                selected_col = st.selectbox("Select categorical column", cat_cols)
                
                # Create distribution plot
                fig = plot_distribution(df, selected_col)
                st.plotly_chart(fig)
                
                # Show distribution by target
                st.subheader(f"{selected_col} by Subscription Status")
                fig = plot_distribution(df, selected_col, "y")
                st.plotly_chart(fig)
                
                # Show value counts
                st.subheader(f"Value Counts for {selected_col}")
                st.write(df[selected_col].value_counts())
        
        elif eda_section == "Correlation Analysis":
            st.subheader("Correlation Analysis")
            
            # Process numeric data for correlation
            if hasattr(st.session_state, 'df_processed'):
                df_processed = st.session_state.df_processed
            else:
                df_processed = df.copy()
                df_processed['y'] = df_processed['y'].map({'yes': 1, 'no': 0})
            
            # Create correlation matrix
            fig = plot_correlation_matrix(df_processed)
            st.plotly_chart(fig)
            
            # Top correlations with target
            st.subheader("Top Correlations with Target (y)")
            
            # Calculate correlations
            numeric_df = df_processed.select_dtypes(include=["int64", "float64"])
            correlations = numeric_df.corr()["y"].sort_values(ascending=False)
            
            # Display correlations
            st.write(correlations)
        
        elif eda_section == "Target Analysis":
            st.subheader("Target Analysis")
            
            # Convert target to binary for analysis
            df_binary = df.copy()
            df_binary['y'] = df_binary['y'].map({'yes': 1, 'no': 0})
            
            # Select feature to analyze against target
            feature_type = st.radio("Select feature type", ["Categorical", "Numerical"])
            
            if feature_type == "Categorical":
                # Get categorical columns
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                cat_cols.remove("y")  # Remove target
                
                # Select column
                selected_col = st.selectbox("Select categorical feature", cat_cols)
                
                # Plot feature-target relationship
                from visualization import plot_feature_target_relationship
                fig = plot_feature_target_relationship(df_binary, selected_col, "y")
                st.plotly_chart(fig)
                
                # Show subscription rates by category
                subscription_rates = df.groupby(selected_col)["y"].apply(
                    lambda x: (x == "yes").mean() * 100
                ).reset_index()
                subscription_rates.columns = [selected_col, "Subscription Rate (%)"]
                subscription_rates = subscription_rates.sort_values("Subscription Rate (%)", ascending=False)
                
                st.write(subscription_rates)
                
            else:  # Numerical
                # Get numerical columns
                num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                
                # Select column
                selected_col = st.selectbox("Select numerical feature", num_cols)
                
                # Plot feature-target relationship
                from visualization import plot_feature_target_relationship
                fig = plot_feature_target_relationship(df_binary, selected_col, "y")
                st.plotly_chart(fig)
                
                # Create bins and show subscription rates by bin
                st.subheader(f"Subscription Rate by {selected_col} Range")
                
                # Create bins
                n_bins = st.slider("Number of bins", 3, 10, 5)
                df_binary[f"{selected_col}_bin"] = pd.cut(df_binary[selected_col], n_bins)
                
                # Calculate subscription rate by bin
                subscription_rates = df_binary.groupby(f"{selected_col}_bin")["y"].mean() * 100
                
                # Plot
                fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
                subscription_rates.plot(kind="bar", ax=ax)
                plt.xlabel(f"{selected_col} Range")
                plt.ylabel("Subscription Rate (%)")
                plt.title(f"Subscription Rate by {selected_col} Range")
                plt.xticks(rotation=45)
                st.pyplot(fig)

# Model Training
elif page == "Model Training":
    st.header("Model Training")
    
    if not st.session_state.data_loaded or 'X_train' not in st.session_state or st.session_state.X_train is None:
        st.warning("Please load and preprocess the data first in the Data Loading section.")
    elif not st.session_state.models_trained:
        st.write("Train Decision Tree and Random Forest models to predict client subscription likelihood.")
        st.write("This may take a few minutes depending on the data size and complexity.")
        
        if st.button("Train Models"):
            try:
                with st.spinner("Training models... This may take a few minutes."):
                    # Train models
                    models, metrics, feature_importances = train_models(
                        st.session_state.X_train,
                        st.session_state.y_train,
                        st.session_state.X_test,
                        st.session_state.y_test
                    )
                    
                    # Store in session state
                    st.session_state.models = models
                    st.session_state.metrics = metrics
                    st.session_state.feature_importances = feature_importances
                    st.session_state.models_trained = True
                    
                    st.success("✅ Models trained successfully!")
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
    else:
        st.success("✅ Models already trained!")
        
        # Show model summary
        st.subheader("Model Summary")
        
        # Display metrics in a table
        metrics = st.session_state.metrics
        
        # Create comparison table
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
            'Decision Tree': [
                f"{metrics['decision_tree']['accuracy']:.4f}",
                f"{metrics['decision_tree']['precision']:.4f}",
                f"{metrics['decision_tree']['recall']:.4f}",
                f"{metrics['decision_tree']['f1']:.4f}",
                f"{metrics['decision_tree']['roc_auc']:.4f}"
            ],
            'Random Forest': [
                f"{metrics['random_forest']['accuracy']:.4f}",
                f"{metrics['random_forest']['precision']:.4f}",
                f"{metrics['random_forest']['recall']:.4f}",
                f"{metrics['random_forest']['f1']:.4f}",
                f"{metrics['random_forest']['roc_auc']:.4f}"
            ]
        })
        
        st.write(metrics_df)
        
        # Plot metrics comparison
        fig = plot_model_metrics_comparison(metrics)
        st.plotly_chart(fig)

# Performance Evaluation
elif page == "Performance Evaluation":
    st.header("Model Performance Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first in the Model Training section.")
    else:
        # Get models and metrics
        metrics = st.session_state.metrics
        
        # Select model
        model_type = st.radio("Select model to evaluate", ["Decision Tree", "Random Forest", "Compare Both"])
        
        if model_type == "Decision Tree":
            # Show confusion matrix
            st.subheader("Confusion Matrix - Decision Tree")
            fig = plot_confusion_matrix(metrics['decision_tree']['confusion_matrix'], "decision_tree")
            st.plotly_chart(fig)
            
            # Detailed metrics
            st.subheader("Detailed Metrics - Decision Tree")
            st.write(f"**Accuracy:** {metrics['decision_tree']['accuracy']:.4f}")
            st.write(f"**Precision:** {metrics['decision_tree']['precision']:.4f}")
            st.write(f"**Recall:** {metrics['decision_tree']['recall']:.4f}")
            st.write(f"**F1 Score:** {metrics['decision_tree']['f1']:.4f}")
            st.write(f"**AUC:** {metrics['decision_tree']['roc_auc']:.4f}")
        
        elif model_type == "Random Forest":
            # Show confusion matrix
            st.subheader("Confusion Matrix - Random Forest")
            fig = plot_confusion_matrix(metrics['random_forest']['confusion_matrix'], "random_forest")
            st.plotly_chart(fig)
            
            # Detailed metrics
            st.subheader("Detailed Metrics - Random Forest")
            st.write(f"**Accuracy:** {metrics['random_forest']['accuracy']:.4f}")
            st.write(f"**Precision:** {metrics['random_forest']['precision']:.4f}")
            st.write(f"**Recall:** {metrics['random_forest']['recall']:.4f}")
            st.write(f"**F1 Score:** {metrics['random_forest']['f1']:.4f}")
            st.write(f"**AUC:** {metrics['random_forest']['roc_auc']:.4f}")
        
        else:  # Compare both
            # Show confusion matrices side by side
            st.subheader("Confusion Matrices")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Decision Tree**")
                fig = plot_confusion_matrix(metrics['decision_tree']['confusion_matrix'], "decision_tree")
                st.plotly_chart(fig)
            
            with col2:
                st.write("**Random Forest**")
                fig = plot_confusion_matrix(metrics['random_forest']['confusion_matrix'], "random_forest")
                st.plotly_chart(fig)
            
            # Show ROC curve comparison
            st.subheader("ROC Curve Comparison")
            
            # Extract metrics for ROC curve
            dt_metrics = metrics['decision_tree']
            rf_metrics = metrics['random_forest']
            
            # Set y_test and y_pred for visualization
            if 'y_test' not in st.session_state:
                st.session_state.y_test = np.array([0, 1])  # Placeholder
            
            dt_metrics['y_pred'] = dt_metrics.get('y_pred', np.array([0.5, 0.5]))  # Placeholder if not available
            rf_metrics['y_pred'] = rf_metrics.get('y_pred', np.array([0.5, 0.5]))  # Placeholder if not available
            
            # Create metrics dict for plotting
            metrics_for_plot = {
                'decision_tree': dt_metrics,
                'random_forest': rf_metrics
            }
            
            # Plot ROC curve
            roc_fig = plot_roc_curve(metrics_for_plot)
            st.plotly_chart(roc_fig)
            
            # Show metrics comparison table
            st.subheader("Metrics Comparison")
            
            comparison_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                'Decision Tree': [
                    f"{metrics['decision_tree']['accuracy']:.4f}",
                    f"{metrics['decision_tree']['precision']:.4f}",
                    f"{metrics['decision_tree']['recall']:.4f}",
                    f"{metrics['decision_tree']['f1']:.4f}",
                    f"{metrics['decision_tree']['roc_auc']:.4f}"
                ],
                'Random Forest': [
                    f"{metrics['random_forest']['accuracy']:.4f}",
                    f"{metrics['random_forest']['precision']:.4f}",
                    f"{metrics['random_forest']['recall']:.4f}",
                    f"{metrics['random_forest']['f1']:.4f}",
                    f"{metrics['random_forest']['roc_auc']:.4f}"
                ],
                'Difference': [
                    f"{metrics['random_forest']['accuracy'] - metrics['decision_tree']['accuracy']:.4f}",
                    f"{metrics['random_forest']['precision'] - metrics['decision_tree']['precision']:.4f}",
                    f"{metrics['random_forest']['recall'] - metrics['decision_tree']['recall']:.4f}",
                    f"{metrics['random_forest']['f1'] - metrics['decision_tree']['f1']:.4f}",
                    f"{metrics['random_forest']['roc_auc'] - metrics['decision_tree']['roc_auc']:.4f}"
                ]
            })
            
            st.write(comparison_df)
            
            # Determine the better model
            better_model = "Random Forest" if metrics['random_forest']['f1'] > metrics['decision_tree']['f1'] else "Decision Tree"
            st.success(f"Based on F1 Score, the **{better_model}** model performs better for this problem.")

# Feature Importance
elif page == "Feature Importance":
    st.header("Feature Importance Analysis")
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first in the Model Training section.")
    else:
        # Get feature importances
        feature_importances = st.session_state.feature_importances
        
        # Select model
        model_type = st.radio("Select model", ["Decision Tree", "Random Forest"])
        model_key = "decision_tree" if model_type == "Decision Tree" else "random_forest"
        
        # Get feature importances
        model_feature_importances = feature_importances[model_key]
        
        # Number of features to display
        n_features = st.slider("Number of top features to display", 5, 20, 10)
        
        # Plot feature importance
        st.subheader(f"Top {n_features} Feature Importances - {model_type}")
        fig = plot_feature_importance(model_feature_importances, model_key, n_features)
        
        if fig is not None:
            st.plotly_chart(fig)
        else:
            st.warning("Feature importance information is not available.")
        
        # Show feature importance table
        if model_feature_importances is not None:
            st.subheader("Feature Importance Table")
            st.write(model_feature_importances.head(n_features))
            
            # Marketing insights
            st.subheader("Marketing Insights")
            
            # Top 5 features
            top_features = model_feature_importances.head(5)['feature'].tolist()
            
            st.markdown("### Key Factors Driving Client Subscription")
            st.markdown(f"""
            The analysis reveals that the following features have the highest impact on client subscription decisions:
            
            1. **{top_features[0] if len(top_features) > 0 else 'N/A'}**
            2. **{top_features[1] if len(top_features) > 1 else 'N/A'}**
            3. **{top_features[2] if len(top_features) > 2 else 'N/A'}**
            4. **{top_features[3] if len(top_features) > 3 else 'N/A'}**
            5. **{top_features[4] if len(top_features) > 4 else 'N/A'}**
            
            ### Recommended Marketing Strategies
            
            Based on these insights, consider the following marketing strategies:
            
            1. **Target Optimization**: Focus marketing efforts on clients with favorable values in the top features
            2. **Campaign Customization**: Design campaigns that address the most influential factors
            3. **Resource Allocation**: Allocate resources based on feature importance
            4. **Messaging Refinement**: Craft messages that emphasize the most important aspects
            """)
        else:
            st.warning("Feature importance information is not available.")

# Footer
st.markdown("---")
st.markdown("""
**Bank Marketing Prediction App** | Developed with Decision Trees and Random Forest
""")
