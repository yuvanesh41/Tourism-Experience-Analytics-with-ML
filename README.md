# Tourism-Experience-Analytics-with-ML

Approach:

Data Cleaning:

Handle missing values in the transaction, user, and city datasets.
Resolve discrepancies in city names or other categorical variables like VisitMode, AttractionTypeId, etc.
Standardize date and time format, ensuring consistency across data.
Handle outliers or any incorrect entries in rating or other columns.

Preprocessing:
Feature Engineering:

Encode categorical variables such as VisitMode, Contenent, Country, and AttractionTypeId.
Aggregate user-level features to represent each user's profile (e.g., average ratings per visit mode).
Join relevant data from transaction, user, city, and attraction tables to create a consolidated dataset.
Normalization: Scale numerical features such as Rating for better model convergence.

Exploratory Data Analysis (EDA):

Visualize user distribution across continents, countries, and regions.
Explore attraction types and their popularity based on user ratings.
Investigate correlation between VisitMode and user demographics to identify patterns.
Analyze distribution of ratings across different attractions and regions.

Model Training:
Regression Task:

Train a model to predict ratings based on user, attractions, transaction features, etc.
Classification Task:
Train a classifier (e.g., Random Forest, LightGBM, or XGBoost) to predict VisitMode based on user and transaction features.

Recommendation Task:

Implement collaborative filtering (using user-item matrix) to recommend attractions based on user ratings and preferences.
Alternatively, use content-based filtering based on attractions' attributes (e.g., location, type).

Model Evaluation:

Evaluate classification model performance using accuracy, precision, recall, and F1-score.
Evaluate regression model using R2, MSE, etc.
Assess recommendation system accuracy using metrics like Mean Average Precision (MAP) or Root Mean Squared Error (RMSE).
Deployment:
Build a Streamlit app that allows users to input their details (location, preferred visit mode) and receive:
A prediction of their visit mode (Business, Family, etc.).
Recommended attractions based on their profile and transaction history.
Display visualizations of popular attractions, top regions, and user segments in the app.
