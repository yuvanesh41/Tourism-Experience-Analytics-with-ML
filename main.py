#importing important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, f1_score, classification_report
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pymysql
import streamlit as st

# Load dataset
df = pd.read_csv("Merged_Dataset.csv")
# Display basic info
df.head()
df.info()
print(df.describe())

# Handling missing values
df.isnull().sum()
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)
df.isnull().sum()

# Count duplicate rows
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Drop duplicates
df_no_duplicates = df.drop_duplicates()
print("\nDataFrame after dropping duplicates:")
print(df_no_duplicates)

#handling outliers
plt.figure(figsize=(5,3))
sns.boxplot(df["Rating"])
plt.title("Outliers in Rating")
plt.show()
df=df[df["Rating"]<=5]

# Encoding categorical variables
label_encoders = {}
for col in ["VisitMode_x","VisitMode_y" ,"ContenentId", "CountryId_x", "AttractionType"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    df.head()

# Feature Engineering
df["User_Visit_Count"] = df.groupby("UserId")["AttractionId"].transform("count")
df["Avg_User_Rating"] = df.groupby("UserId")["Rating"].transform("mean")
df["Attraction_Popularity"] = df.groupby("AttractionId")["Rating"].transform("mean")
print("DataFrame after Feature Engineering:")
print(df)

# Scaling numerical features
#  Normalization
scaler = MinMaxScaler()
df[["Avg_User_Rating", "Attraction_Popularity"]] = scaler.fit_transform(df[["Avg_User_Rating", "Attraction_Popularity"]])
scaler = StandardScaler()
df[['Rating']] = scaler.fit_transform(df[['Rating']])
scaler = MinMaxScaler()
df[["Avg_User_Rating", "Attraction_Popularity"]] = scaler.fit_transform(df[["Avg_User_Rating", "Attraction_Popularity"]])
print("DataFrame after Scaling:")
print(df)

# EDA
st.subheader("📊 Exploratory Data Analysis")

fig1 = px.histogram(df, x="Rating", title="Rating Distribution")
st.plotly_chart(fig1)

fig2 = px.box(df, x="VisitMode_x", y="Rating", title="Visit Mode vs Rating")
st.plotly_chart(fig2)

fig3 = px.bar(df["AttractionType"].value_counts(), title="Popular Attraction Types")
st.plotly_chart(fig3)

fig, axes = plt.subplots(2, 5, figsize=(5, 3))
sns.histplot(df['Rating'], kde=True, ax=axes[0,0])
sns.countplot(x='VisitMode_x', data=df, ax=axes[0,1])
sns.boxplot(x='VisitMode_y', y='Rating', data=df, ax=axes[0,2])
sns.scatterplot(x='VisitYear', y='Rating', data=df, ax=axes[0,4])
sns.violinplot(x='AttractionType', y='Rating', data=df, ax=axes[1,0])
sns.barplot(x='ContenentId', y='Rating', data=df, ax=axes[1,1])
sns.kdeplot(df['Rating'], shade=True, ax=axes[1,2])
sns.lineplot(x='UserId', y='Rating', data=df, ax=axes[1,3])
sns.boxplot(x='AttractionType', y='Rating', data=df, ax=axes[1,4])
plt.show()

# Splitting dataset for models
X = df.drop(columns=['Rating'])  # Features (Independent Variables)
y = df['Rating']  # Target (Dependent Variable)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Machine learning models
st.subheader("🤖 Machine Learning Models")

# 1)Regression Model (Predicting Ratings)

X_reg = df[["ContenentId", "CountryId_x", "VisitMode_x","VisitMode_y" ,"VisitMonth", "AttractionTypeId"]]
y_reg = df["Rating"]
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
model_reg = RandomForestRegressor()
model_reg.fit(X_train, y_train)
y_pred_reg = model_reg.predict(X_test)
st.write(f"Regression Model RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_reg))}")

# 2)Classification Model (Predicting Visit Mode)

X_clf = df[["ContenentId", "CountryId_x", "User_Visit_Count", "Attraction_Popularity"]]
y_clf = df["VisitMode_y"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
model_clf = RandomForestClassifier()
model_clf.fit(X_train, y_train)
y_pred_clf = model_clf.predict(X_test)
st.write(f"Classification Model Accuracy: {accuracy_score(y_test, y_pred_clf)}")
st.write(classification_report(y_test, y_pred_clf))

#Recommendation System

st.subheader("🎯Recommendations to user-Suggested based on their previous activities")


#1) ---------------- CONTENT-BASED FILTERING ----------------
def recommend_attractions_content_based(attraction_id, df, n=5):
    """ Recommends attractions similar to the given attraction using Content-Based Filtering """
    attraction_features = df[df["AttractionId"] == attraction_id][["AttractionType", "UserId"]].values
    knn = NearestNeighbors(n_neighbors=n, metric="cosine")
    knn.fit(df[["AttractionType", "UserId"]])
    distances, indices = knn.kneighbors(attraction_features)
    recommendations = df.iloc[indices[0]]["Attraction"].tolist()
    return recommendations

#2) ---------------- COLLABORATIVE FILTERING (USER-BASED) ----------------
def recommend_attractions_collaborative(user_id, df, n=5):
    """ Recommends attractions based on similar users' ratings using Collaborative Filtering """
    user_item_matrix = df.pivot_table(index="UserId", columns="AttractionId", values="Rating", aggfunc="mean")

    user_sparse_matrix = csr_matrix(user_item_matrix.fillna(0))

    # Finding Similar Users
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(user_sparse_matrix)
    user_index = user_item_matrix.index.get_loc(user_id)

    distances, indices = knn.kneighbors(user_sparse_matrix[user_index], n_neighbors=n+1)
    similar_users = indices.flatten()[1:]

    # Get attractions liked by similar users
    recommendations = []
    for sim_user in similar_users:
        top_attractions = user_item_matrix.iloc[sim_user].sort_values(ascending=False).index[:n]
        recommendations.extend(top_attractions)

    return list(set(recommendations))[:n]


# MySQL Integration
conn = pymysql.connect(host='localhost', user='root', password='vijay45', database='tourism_db')
cursor = conn.cursor()
df = df[['TransactionId', 'UserId', 'VisitYear', 'VisitMonth', 'VisitMode_x', 'AttractionId', 'Rating',
         'ContenentId', 'RegionId', 'CountryId_x', 'CityId_x', 'AttractionCityId', 'AttractionTypeId', 
         'Attraction', 'AttractionAddress', 'CityId_y', 'CityName', 'CountryId_y', 'AttractionType', 
         'VisitModeId', 'VisitMode_y']]

# Convert NaN values to None (so MySQL/MariaDB can store them as NULL)
df = df.replace({np.nan: None})

# Convert DataFrame to a list of tuples
data_tuples = [tuple(x) for x in df.to_numpy()]

# Insert multiple rows at once using executemany()
insert_query = """
    INSERT INTO tourism_db.merged_dataset(
        TransactionId, UserId, VisitYear, VisitMonth, VisitMode_x, AttractionId, Rating, 
        ContenentId, RegionId, CountryId_x, CityId_x, AttractionCityId, AttractionTypeId, 
        Attraction, AttractionAddress, CityId_y, CityName, CountryId_y, AttractionType, 
        VisitModeId, VisitMode_y)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE Rating = VALUES(Rating);
"""
cursor.executemany(insert_query, data_tuples)
conn.commit()

print("✅ Bulk data inserted into tourism_data successfully!")

# SQL Queries for Analysis
st.subheader("📌 SQL Insights")
query = """
SELECT Attraction, COUNT(*) AS VisitCount
FROM tourism_db.merged_dataset
GROUP BY Attraction
ORDER BY VisitCount DESC
LIMIT 5;
"""
cursor.execute(query)

# Fetch results
top_attractions = cursor.fetchall()

# Display results
print("🏆 Top 5 Most Visited Attractions:")
for attraction in top_attractions:
    print(f"Attraction: {attraction[0]}, Visits: {attraction[1]}")
# Count of Visits Based on Visit Mode
query = """
SELECT VisitMode_x, COUNT(*) AS VisitCount
FROM tourism_db.merged_dataset
GROUP BY VisitMode_x
ORDER BY VisitCount DESC;
"""
cursor.execute(query)

# Fetch results
visit_mode_counts = cursor.fetchall()

# Display results
print("🗂️ Visit Mode Distribution:")
for mode in visit_mode_counts:
    print(f"Visit Mode: {mode[0]}, Visits: {mode[1]}")
#Average Rating of Attractions
query = """
SELECT Attraction, ROUND(AVG(Rating),2) AS AvgRating
FROM tourism_db.merged_dataset
GROUP BY Attraction
ORDER BY AvgRating DESC
LIMIT 10;
"""
cursor.execute(query)
st.write(pd.DataFrame(cursor.fetchall()))

st.title("Tourism Experience Analytics")
st.sidebar.title("🌍 Travel Experience Dashboard")
option = st.sidebar.selectbox("Choose an analysis", ["EDA", "Prediction", "Recommendation"])
if option == "EDA":
    st.write("📊 Select an EDA chart from above!")
elif option == "Prediction":
    user_input = st.text_input("Enter User ID for prediction")
    if user_input:
        pred_rating = model_reg.predict([[1, 2, 0, 10, 7, 63]])[0]
        st.write(f"Predicted Rating: {pred_rating}")
elif option == "Recommendation":
    attraction_input = st.text_input("Enter Attraction ID")
st.sidebar.title("🔍 Get Recommendations")
option = st.sidebar.radio("Choose Recommendation Type", ["Content-Based", "Collaborative Filtering"])

if option == "Content-Based":
    attraction_input = st.number_input("Enter Attraction ID:", min_value=int(df["AttractionId"].min()), max_value=int(df["AttractionId"].max()))
    if st.button("Get Recommendations"):
        st.write(f"Recommended attractions: {recommend_attractions_content_based(attraction_input, df)}")

elif option == "Collaborative Filtering":
    user_input = st.number_input("Enter User ID:", min_value=int(df["UserId"].min()), max_value=int(df["UserId"].max()))
    if st.button("Get Recommendations"):
        st.write(f"Recommended attractions for User ID {user_input}: {recommend_attractions_collaborative(user_input, df)}")
