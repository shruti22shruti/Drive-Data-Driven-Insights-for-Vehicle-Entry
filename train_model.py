from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "JPN DATA.xlsx"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "drive_logistic_model.pkl"

# Columns used by the model
FEATURES = [
    "AGE_CAR SEGMENT",
    "AGE_GROUP",
    "INCOME_GROUP",
    "GENDER_CODE",
]

TARGET = "PURCHASE"


def main() -> None:
    """Train and evaluate the DRIVE logistic regression model."""

    # Read the training and testing sheets from the processed Japan workbook
    training_data = pd.read_excel(DATA_FILE, sheet_name=" Training Data")
    testing_data = pd.read_excel(DATA_FILE, sheet_name="Testing Data")

    # Select only the required columns
    x_train = training_data[FEATURES]
    y_train = training_data[TARGET]

    x_test = testing_data[FEATURES]
    y_test = testing_data[TARGET]

    # Create and train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    # Generate test predictions
    predicted_labels = model.predict(x_test)
    predicted_probabilities = model.predict_proba(x_test)[:, 1]

    # Calculate validation metrics
    accuracy = accuracy_score(y_test, predicted_labels)
    auc = roc_auc_score(y_test, predicted_probabilities)
    matrix = confusion_matrix(y_test, predicted_labels)

    print("\nDRIVE MODEL TRAINING COMPLETED")
    print("--------------------------------")
    print(f"Training records: {len(training_data):,}")
    print(f"Testing records:  {len(testing_data):,}")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"AUC:              {auc:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(classification_report(y_test, predicted_labels))

    print("\nModel Coefficients:")
    print(f"Intercept: {model.intercept_[0]:.6f}")

    for feature, coefficient in zip(FEATURES, model.coef_[0]):
        print(f"{feature}: {coefficient:.6f}")

    # Create the model folder if required and save the trained model
    MODEL_DIR.mkdir(exist_ok=True)

    model_package = {
        "model": model,
        "features": FEATURES,
        "cutoff": 0.5,
    }

    joblib.dump(model_package, MODEL_FILE)

    print(f"\nModel saved successfully at:\n{MODEL_FILE}")


if __name__ == "__main__":
    main()