# ⚡ City Energy Consumption Analysis & Prediction System

> A Python-based data analysis and machine learning project for analyzing electricity consumption across multiple city zones and predicting next-day energy consumption.

---

## 📌 Project Overview

The **City Energy Consumption Analysis & Prediction System** is an end-to-end Python project designed to analyze electricity consumption patterns across different city zones.

The project generates a synthetic one-year electricity consumption dataset, performs data cleaning and exploratory analysis, creates visualizations, and uses a **Random Forest Regression** model to predict next-day energy consumption.

The project demonstrates practical applications of:

- Python programming
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature engineering
- Machine learning
- Model evaluation
- Energy consumption forecasting

---

## 🎯 Objectives

The main objectives of this project are:

1. Generate a realistic synthetic electricity consumption dataset.
2. Clean and validate the generated data.
3. Analyze electricity consumption trends.
4. Compare energy consumption across different city zones.
5. Visualize important consumption patterns.
6. Engineer features for machine learning.
7. Train a Random Forest regression model.
8. Predict next-day electricity consumption.
9. Evaluate model performance using Mean Absolute Error (MAE).
10. Provide an interactive prediction workflow.

---

## 🏙️ Project Features

### 📊 Data Generation

The project generates a synthetic dataset containing:

- **365 days** of observations
- **5 city zones**
- **1,825 total readings**

The dataset represents daily electricity consumption for different city zones.

### 🧹 Data Cleaning

The project performs data validation and cleaning to ensure:

- Correct data types
- Valid dates
- No unnecessary duplicate records
- Valid consumption values
- Consistent zone information

### 📈 Exploratory Data Analysis

The project analyzes:

- Monthly electricity consumption trends
- Zone-level consumption
- Minimum consumption
- Maximum consumption
- Average consumption
- Event vs. non-event consumption
- Relationships between numerical variables

### 📉 Data Visualization

The project generates visualizations including:

- Monthly energy consumption trend
- Correlation heatmap
- Event vs. non-event consumption comparison

Generated plots are stored in the `outputs/` directory.

### 🤖 Machine Learning

A **Random Forest Regressor** is used to predict next-day energy consumption.

The model uses engineered features based on historical consumption patterns.

### 📏 Model Evaluation

The model is evaluated using:

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between actual and predicted energy consumption.

A lower MAE indicates better prediction performance.

---

## 🗂️ Project Structure

```text
city-energy-consumption-analysis2/
│
├── city_energy_analysis.py
│
├── City_Energy_Consumption_Analysis.ipynb
│
├── README.md
│
├── requirements.txt
│
├── data/
│   └── city_energy.csv
│
└── outputs/
    ├── monthly_energy_trend.png
    ├── correlation_heatmap.png
    └── event_vs_non_event.png
