# Drive-Data-Driven-Insights-for-Vehicle-Entry
This project analyses customer purchasing behaviour using the Japan automobile dataset to build a classification model that predicts whether a customer is likely to buy a new vehicle. The insights are then applied to Indian customer data to estimate market potential and identify prospective customers.

# DRIVE — Data-Driven Insights for Vehicle Entry

**A Machine Learning-Based Customer Purchase Prediction and Analytics Application**

## 1. Project Overview

DRIVE (Data-Driven Insights for Vehicle Entry) is a data analytics and machine learning project developed as part of the Internshala Data Science with AI internship.

The project explores the potential for an automobile company operating in Japan to expand into the Indian market. It uses historical Japanese customer data to develop a Logistic Regression model and applies the trained model to Indian customer data to estimate potential vehicle buyers.

The project was further extended into an interactive web application using **Python and Streamlit**, allowing users to make individual predictions, perform bulk predictions, explore insights, and download prediction results.

## 2. Objectives

- Analyze historical customer data to identify purchasing patterns.
- Perform data preprocessing and feature engineering.
- Develop a Logistic Regression classification model.
- Estimate potential vehicle buyers using Indian customer data.
- Build an interactive application for customer-level and bulk predictions.
- Present insights through data visualizations and dashboards.

## 3. Project Workflow

1. **Data Collection:** Worked with historical Japanese customer data and Indian customer data.
2. **Data Preparation:** Cleaned and prepared the datasets using Excel and Python.
3. **Exploratory Data Analysis:** Examined customer characteristics and purchasing patterns.
4. **Feature Engineering:** Created meaningful features and transformed input variables for model training.
5. **Model Development:** Trained a Logistic Regression classification model.
6. **Model Evaluation:** Evaluated model performance using classification metrics.
7. **Prediction:** Applied the trained model to the Indian customer dataset.
8. **Application Development:** Built an interactive Streamlit application for predictions and analytics.

## 4. Technologies Used

- **Python** – Data processing and application development
- **Pandas** – Data manipulation and analysis
- **Scikit-learn** – Machine learning and model evaluation
- **Joblib** – Model serialization and loading
- **Streamlit** – Interactive web application
- **Plotly** – Interactive data visualizations
- **Excel** – Data analysis and preprocessing
- **Tableau** – Data visualization and dashboard development

## 5. Key Features

- **Individual Prediction:** Predicts the likelihood of a customer purchasing a vehicle based on input details.
- **Bulk Prediction:** Processes multiple customer records in one go.
- **Interactive Dashboard:** Displays useful insights and prediction summaries.
- **Downloadable Results:** Allows users to download prediction outputs.
- **About DRIVE:** Provides an overview of the project and its purpose.

## 6. Key Finding

The model estimated **41,344 potential buyers out of 70,000 Indian customer records**, representing approximately 59.06% of the dataset.

This is a model-generated estimate based on the available data and the selected classification threshold of 0.5. It should not be interpreted as a guarantee of actual purchases or sales.

## 7. Project Structure

```text
DRIVE/
│
├── app.py
├── train_model.py
├── requirements.txt
│
├── data/
│   ├── processed_india_data.csv
│   └── processed_japan_data.csv
│
└── model/
    └── drive_logistic_model.pkl
```

*Note: The file names above are illustrative. Adjust them to match the actual files in your repository.*

## 8. How to Run the Application

### Prerequisites
- Python installed
- Git (optional)

### Installation

1. Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

2. Navigate to the project directory:

```bash
cd DRIVE
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

5. Open the local URL displayed in the terminal to access the application.

## 9. Limitations

- The model is trained using Japanese customer data and applied to Indian customer data, which may not fully represent differences in consumer behaviour between the two markets.
- Predictions depend on the quality and characteristics of the available datasets.
- Model-generated potential buyers should not be treated as confirmed customers or actual sales.
- The application is a prototype and is not connected to a live customer database or production system.

## 10. Future Improvements

- Incorporate more India-specific customer and market information.
- Explore additional machine learning algorithms and compare their performance.
- Improve prediction reliability through more representative datasets.
- Deploy the application online for wider accessibility.
- Integrate updated data sources for more dynamic analysis.

## 11. Acknowledgements

This project was developed as part of the **Internshala Data Science with AI internship**, based on a business case involving ABG Motors and its potential expansion into India.

---

**DRIVE — Turning customer data into insights for informed business decisions.**
