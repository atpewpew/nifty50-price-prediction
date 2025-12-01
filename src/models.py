"""
ML Models for Price Direction Prediction

This module contains training and evaluation functions for comparing
multiple ML models as required by the assignment.

Models implemented:
1. Random Forest Classifier - Ensemble of decision trees with bagging
2. Histogram Gradient Boosting Classifier - Fast gradient boosting (similar to LightGBM)
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier.
    
    Random Forest uses bagging (bootstrap aggregating) with multiple decision trees.
    Good baseline model that's robust to overfitting.
    
    Args:
        X_train: Training features
        y_train: Training labels (0 = price down, 1 = price up)
    
    Returns:
        Trained RandomForestClassifier model
    """
    print("  Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,      # Number of trees
        max_depth=15,          # Limit depth to prevent overfitting
        min_samples_split=10,  # Minimum samples to split a node
        min_samples_leaf=5,    # Minimum samples in leaf node
        random_state=42,       # Reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """
    Train a Histogram Gradient Boosting Classifier.
    
    HistGradientBoosting is similar to LightGBM/XGBoost.
    Uses sequential boosting where each tree corrects previous errors.
    Natively handles NaN values and is very fast.
    
    Args:
        X_train: Training features
        y_train: Training labels (0 = price down, 1 = price up)
    
    Returns:
        Trained HistGradientBoostingClassifier model
    """
    print("  Training Gradient Boosting...")
    model = HistGradientBoostingClassifier(
        max_iter=200,          # Number of boosting iterations
        learning_rate=0.05,    # Step size shrinkage
        max_depth=10,          # Maximum tree depth
        random_state=42,       # Reproducibility
        early_stopping=True    # Stop if validation score doesn't improve
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and return performance metrics.
    
    Args:
        model: Trained sklearn classifier
        X_test: Test features
        y_test: True test labels
        model_name: Name for display purposes
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and predictions
    """
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    
    return {
        'name': model_name,
        'predictions': predictions,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def train_and_compare_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare their performance.
    
    This function trains both Random Forest and Gradient Boosting models,
    evaluates them on the test set, and returns the best performing model
    based on accuracy.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: True test labels
    
    Returns:
        tuple: (best_model, best_results, all_results)
            - best_model: The trained model with highest accuracy
            - best_results: Metrics dict for the best model
            - all_results: List of metrics dicts for all models
    """
    print("\n" + "="*60)
    print("MODEL TRAINING & COMPARISON")
    print("="*60)
    
    # Train both models
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Evaluate both models
    print("\n--- Evaluating Models on Test Set ---")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    gb_results = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    
    all_results = [rf_results, gb_results]
    models = {'Random Forest': rf_model, 'Gradient Boosting': gb_model}
    
    # Print comparison table
    print("\n" + "-"*60)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*60)
    for result in all_results:
        print(f"{result['name']:<25} {result['accuracy']:<12.4f} {result['precision']:<12.4f} {result['recall']:<12.4f}")
    print("-"*60)
    
    # Select best model based on accuracy
    best_results = max(all_results, key=lambda x: x['accuracy'])
    best_model = models[best_results['name']]
    
    print(f"\n[SELECTED] Best Model: {best_results['name']} (Accuracy: {best_results['accuracy']:.4f})")
    
    return best_model, best_results, all_results