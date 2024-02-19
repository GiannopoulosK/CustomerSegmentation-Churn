# **Welcome to my project- Customer Segmentation and Churn Prediction** 
### In this project, i utilized Machine Learning techniques to achieve the following results:
## **Clustering:**    
| Metric                  | Value                  |
|-------------------------|------------------------| 
| Inertia                 | 4278.431585688422      |  
| Silhouette Score        | 0.3087673678611763     | 
| Calinski Harabasz Score | 2978.4423628096406     |
| Davies Bouldin Score    | 1.3723072336824869     |                                                                                

**Churn Prediction:**
| Model                     |   Class 1 Recall |   Class 1 F1 |   Accuracy |
|---------------------------|------------------|--------------|------------|
| Base Logistic Regression  |             0.84 |         0.65 |       0.85 |
| Base Random Forest        |             0.83 |         0.85 |       0.95 |
| Base XGBoost              |             0.89 |         0.91 |       0.97 |
| Base LightGBM             |             0.89 |         0.9  |       0.97 |
| Best Logistic Regression  |             0.91 |         0.46 |       0.65 |
| Best Random Forest        |             0.84 |         0.86 |       0.96 |
| Best XGBClassifier        |             0.92 |         0.87 |       0.96 |
| Best LightGBMClassifier   |             0.89 |         0.86 |       0.95 |
| Best XGBClassifier Tuned  |             0.89 |         0.86 |       0.95 |
| Best LGBMClassifier Tuned |             0.89 |         0.87 |       0.96 |
| Ensemble Base Models      |             0.89 |         0.9  |       0.97 |
| Ensemble Tuned Models     |             0.9  |         0.89 |       0.96 |
| Ensemble XGB Models       |             0.89 |         0.9  |       0.97 |

# Let's see how i did it!

## Tools and Libraries Used:
- ### Pandas
- ### NumPy
- ### SciPy
- ### Matplotlib
- ### Seaborn
- ### Scikit-learn
- ### Imbalanced-learn
- ### XGBoost
- ### LightGBM

# Data: 
### For this project, i used data  directly from kaggle without doing any data collection myself. Below are the links for the dataset:

Dataset URL: [Original Dataset](https://zenodo.org/records/4322342#.Y8OsBdJBwUE)

Kaggle Dataset URL: [Dataset on Kaggle](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m)

# EDA:
### During Exploratory Data Analysis, i delved deep into the data performing univariate, biviriate and multivariate analysis, in which i identified a lot of outliers in four of the columns, as seen below:
![Outlier Dashboard](outlier_dashboard.png)

## Outlier Detection and Handling:
### For detecting the outliers, i used an ensemble technique using:
- #### Modified Zscore
- #### IQR
- #### Isolation Forest
### and combining their results to get more a robust output.
![Isolation Forest Example](isolation_forest_example.png)

#### Zscore was not used because of the high skewness these variables had.

### After running the models with and without outliers, it turned out performance was better if no outlier handling was done.









