import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

# Streamlit app
st.title("Diabetes Classification using LDA and SVM")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the data
    st.write("### Data Preview")
    st.write(data.head())
    
    # Check if 'target' column exists
    if 'target' not in data.columns:
        st.error("The dataset must include a 'target' column for labels.")
    else:
        # Split features and target
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dimensionality Reduction using LDA
        lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = lda.fit_transform(X_scaled, y)
        
        # Feature importance
        feature_importance = np.abs(lda.scalings_.flatten())
        feature_names = X.columns.tolist()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        # Display feature importance
        st.write("### Feature Importance from LDA")
        st.write(importance_df)
        
        # Visualize feature importance
        green_to_red = LinearSegmentedColormap.from_list('GreenToRed', ['#FF0000', '#00FF00'])
        plt.figure(figsize=(12, 6))
        norm = Normalize(vmin=importance_df['Importance'].min(), vmax=importance_df['Importance'].max())
        colors = [green_to_red(norm(value)) for value in importance_df['Importance']]
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette=colors)
        plt.title('Feature Importance from LDA for Diabetes Classification')
        plt.xlabel('LDA Coefficient (Absolute Value)')
        plt.ylabel('Feature')
        plt.tight_layout()
        st.pyplot(plt)
        
        # Split the LDA-transformed data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)
        
        # Standardize the LDA-transformed data
        X_train_scaled_lda = scaler.fit_transform(X_train)
        X_test_scaled_lda = scaler.transform(X_test)
        
        # SVM model with RandomizedSearchCV
        param_dist = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1],
            'kernel': ['rbf', 'linear', 'poly']
        }
        svm = SVC()
        random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=5,
                                           cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        random_search.fit(X_train_scaled_lda, y_train)
        
        # Best SVM model
        best_svm = random_search.best_estimator_
        st.write("### Best Parameters")
        st.write(random_search.best_params_)
        
        # Predictions
        y_pred = best_svm.predict(X_test_scaled_lda)
        
        # Classification report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
        
        # Performance metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        st.write(f"**Accuracy:** {acc:.4f}")
        st.write(f"**Precision:** {prec:.4f}")
        st.write(f"**Recall:** {rec:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        st.pyplot(plt)
