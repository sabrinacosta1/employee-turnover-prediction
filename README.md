# employee-turnover-prediction
 ## Case Study #1: Key Learning Outcomes

This project demonstrates how data science can be applied in a human resources context to predict employee turnover. It covers various machine learning techniques, including logistic regression and random forest classifiers, and emphasizes important data science concepts such as exploratory data analysis, model evaluation, and neural networks.

### Learning Objectives
- Understand how to leverage data science to reduce employee turnover and improve HR practices.
- Learn the theory behind logistic regression and random forest classifiers.
- Train machine learning models using Scikit-Learn (Logistic Regression and Random Forest).
- Apply the sigmoid function to generate probabilities for classification tasks.
- Load, manipulate, and process data using Pandas.
- Develop and apply custom functions to Pandas DataFrames.
- Perform exploratory data analysis (EDA) using Matplotlib and Seaborn.
- Visualize data with Kernel Density Estimation (KDE), box plots, and count plots.
- Convert categorical variables into dummy variables for model training.
- Split the dataset into training and testing sets using Scikit-Learn.
- Understand and apply artificial neural networks for classification tasks.
- Evaluate classification models with confusion matrix and classification reports.
- Understand key classification metrics such as precision, recall, and F1-score.

## Overview
Employee attrition is a major challenge for businesses, affecting productivity, morale, and the bottom line. This project uses **machine learning** and **data analysis** to predict which employees are most likely to leave, providing valuable insights to improve retention strategies.

By analyzing HR data, we explore key factors contributing to employee attrition, build predictive models, and visualize critical patterns. The project combines **statistical analysis**, **classification models**, and **deep learning techniques**, making it a comprehensive example of how data science can address real-world business problems.

It was created while following the guidance of Dr. Ryan Ahmed, Ph.D., MBA.


### Problem Statement

Hiring and retaining employees is a complex, resource-intensive process. The costs associated with employee turnover can significantly impact a company's revenue and productivity.

- Small business owners spend 40% of their working hours on non-revenue generating tasks like hiring.
- Companies spend 15%-20% of the employee's salary on recruitment.
- On average, a company loses between 1% and 2.5% of its total revenue due to the time it takes to bring a new hire up to speed.
- Hiring a new employee costs an average of $7,645 for companies with 0-500 employees.
- It takes 52 days on average to fill a position.

**Source**: https://toggl.com/blog/cost-of-hiring-an-employee

### Business Case

As a data scientist at a multinational corporation, you have been tasked with developing a predictive model to help the HR department identify employees who are most likely to leave the company. This predictive model will allow the HR team to take proactive measures to reduce turnover.

### Dataset Features (Sample)
- JobInvolvement
- Education
- JobSatisfaction
- PerformanceRating
- RelationshipSatisfaction
- WorkLifeBalance

**Data Source**: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Dataset Features
The dataset contains key employee metrics, such as:
- **Categorical**: JobRole, MaritalStatus, EducationField, Gender, etc.
- **Numerical**: MonthlyIncome, DistanceFromHome, TotalWorkingYears, YearsWithCurrentManager, etc.
- **Target Variable**: `Attrition` (1 = Left, 0 = Stayed).

**Data Source**: [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Project Structure

The project consists of the following main components:
1. **Data Loading and Preprocessing**: Load the dataset, clean it, and handle categorical and missing values.
2. **Exploratory Data Analysis (EDA)**: Visualize the data to identify trends and insights using Matplotlib and Seaborn.
3. **Feature Engineering**: Convert categorical variables to dummy variables and prepare the data for model training.
4. **Modeling**: Train logistic regression and random forest classifiers using Scikit-Learn, and apply artificial neural networks.
5. **Model Evaluation**: Evaluate the models using confusion matrix, classification reports, and key metrics such as precision, recall, and F1-score.

## Project Highlights

- **Real-World Problem**: Focused on reducing employee attrition, a $7,645 average cost per hire challenge for businesses.
- **Key Technologies**: Python, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow.
- **Algorithms Used**:
  - Logistic Regression
  - Random Forest Classifier
  - Deep Learning (Neural Networks)
- **Insights Gained**:
  - Relationship between job satisfaction, work-life balance, and attrition.
  - Importance of age, income, distance from home, and other factors.
- **Visualization-Driven**: Extensive EDA (Exploratory Data Analysis) with heatmaps, KDE plots, and feature importance visualizations.

## Installation & Usage
### Prerequisites
Ensure you have Python 3.x and the following libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/sabrinacosta1/employee-Turnover-Prediction.git
   cd Employee Turnover Prediction

## Usage

- Load Data: Add the dataset file to the project directory.
- Run Analysis: Open and run the notebook or script.
- Evaluate Models: Test predictions and visualize the results.


## Results

### Details of the Results
**Class 0 (Employees Who Stayed):**
- **Precision**: 89% → When the model predicts that someone stayed, it is correct 89% of the time.
- **Recall**: 93% → The model successfully identifies 93% of employees who actually stayed.
- **F1-Score**: 91% → Shows a solid balance between precision and recall.

**Class 1 (Employees Who Left):**
- **Precision**: 55% → Only 55% of the predictions for employees who left are correct.
- **Recall**: 43% → The model identifies only 43% of employees who actually left.
- **F1-Score**: 48% → The performance for this class is significantly lower.

**Overall Accuracy:**
- **85%** → Despite the good overall value, this is influenced by the fact that Class 0 (employees who stayed) is far more frequent (class imbalance).


### Critical Analysis
**Class Imbalance:**
- The number of employees who stayed (307) is much higher than the number of employees who left (61). This causes the model to favor Class 0, leading to a lower recall for Class 1.

**Impact of Low Recall for Class 1:**
- Since the goal of the project is to identify employees at risk of leaving, the recall for Class 1 (43%) needs to be improved. A low recall means the model is missing many employees who actually left.

---

### Next Steps to Improve the Model
**Class Balancing Techniques:**
1. **Oversampling**: Increase the number of examples in the minority class (e.g., SMOTE - Synthetic Minority Oversampling Technique).
2. **Undersampling**: Reduce the number of examples in the majority class.

**Adjusting the Threshold:**
- Instead of using the default threshold (0.5), adjust it to prioritize identifying Class 1 (employees who leave), even if it reduces precision.

**Using Algorithms Sensitive to Imbalance:**
- Algorithms like **XGBoost** or **LightGBM** have specific parameters to handle class imbalance effectively.

**Trying Different Metrics:**
- Replace accuracy with the **F1-score for Class 1**, or use **AUC-ROC** for a more robust analysis.

**Hyperparameter Tuning:**
- Perform a search for better parameters (e.g., using GridSearchCV or RandomizedSearchCV) to improve the model's performance for Class 1.
