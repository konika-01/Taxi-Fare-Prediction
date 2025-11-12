### Taxi Fare Price Prediction — Regression Analysis
---
### `Abstract`

Accurately predicting taxi fares is essential for optimizing pricing strategies, improving customer satisfaction, and assisting drivers in estimating expected trip earnings. Taxi fare prices depend on multiple factors such as trip distance, duration, pickup and drop locations, and time of travel.

This project leverages supervised machine learning regression techniques to predict taxi fares based on trip and location-related data.

---
### `Problem Statement`

To predict the taxi fare amount using trip-related features such as pickup/drop coordinates, trip distance, passenger count, and timestamp information. The goal is to minimize prediction error and improve fare estimation accuracy.

---
### `Dataset`

Dataset: CSV file added 

Target Variable: fare_amount

Features include:

pickup_datetime, pickup_longitude, pickup_latitude

dropoff_longitude, dropoff_latitude, passenger_count and more.

Derived feature: Haversine Distance (trip distance between pickup and drop points)

---
#   `Tools & Libraries Used`

1. Python
2. Pandas — Data cleaning and transformation
3. NumPy — Numerical computation
4. Matplotlib / Seaborn — Data visualization and analysis
5. Folium — Map-based visualization of pickup and drop points
6. Scikit-learn (sklearn) — Model building and evaluation
7. Pickle — Model serialization and saving

---
### `Data Preprocessing Steps`

Handled missing values — Removed or imputed NaN entries.

Removed duplicates — Ensured no duplicate trips existed.

Ensured data consistency — Validated coordinates, passenger counts, and fare ranges.

Corrected data types — Converted columns like pickup_datetime to datetime.

Feature Engineering

Created Haversine Distance to represent trip distance using latitude and longitude.

Extracted time-based features (hour, day, month) from pickup_datetime.

Renamed columns for clarity.

Univariate and Multivariate Analysis

Visualized distribution of fares, passenger counts, and distances.

Analyzed correlations and feature relationships using heatmaps and pair plots.

---
### `Model Building and Model Performance`

**Linear Regression (Baseline Model)**

Train R² : 0.625 
Test R² : 0.627
MAE : 3.116
MSE : 31.033
RMSE : 5.571
Observation: Basic linear relationship, limited fit


**Decision Tree Regression**


Train R² : 0.817 
Test R² : 0.793
MAE : 2.189
MSE : 17.198
RMSE : 4.147
Observation: Better non-linear learning, slight overfitting

**Random Forest Regression (Optimized using RandomizedSearchCV)**

Train R² : 0.970
Test R² : 0.805
MAE : 2.129
MSE : 16.235
RMSE : 4.029
Observation: Strong generalization, best performance

**Train-Test Split: 70% training and 30% testing**
				
**Hyperparameter Tuning**

Performed RandomizedSearchCV to optimize Random Forest parameters:

print(rs.best_score_)
print(rs.best_params_)

Results:

Best Score: -16.0137

Best Parameters:

{'n_estimators': 85, 'min_samples_split': 12, 'max_depth': 9}


These parameters improved model generalization and reduced variance without overfitting.

--- 
### `Model Saving`

Final Random Forest Model was saved using Pickle for deployment:

import pickle
pickle.dump(rf_model, open('taxi_fare_model.pkl', 'wb'))


This allows easy loading for future predictions or integration into web apps.

---
### `Visualization Insights`

Folium Maps: Displayed pickup and drop-off clusters across city areas.

Distance vs Fare: Clear positive relationship observed.

Outlier Detection: Trips with unrealistic coordinates or extremely high fares were filtered out.

Time of day, traffic, and weather conditions helps in better predictions

--- 

### `Future Improvements`

Use Gradient Boosting or XGBoost for further accuracy improvements.

Build a Streamlit/Flask dashboard to allow real-time fare predictions.

Implement cross-validation for more robust model evaluation.
