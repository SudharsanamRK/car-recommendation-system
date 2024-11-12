# Car Recommendation System

The **Car Recommendation System** is an innovative solution built using machine learning that helps users find the most suitable cars based on their specific preferences, such as budget, fuel type, and transmission. The system utilizes **KMeans clustering** to categorize cars into different segments, like **Budget-Friendly**, **Mid-Range**, and **Luxury**, making it easy for users to explore cars that match their requirements. This recommendation system not only provides a personalized experience but also includes data visualizations to help users make informed decisions.

---

## Problem Statement

Many car buyers face difficulty in narrowing down their options based on personal preferences like budget, fuel type, transmission type, and more. This project aims to create a **Car Recommendation System** that helps users get personalized car suggestions in real-time based on their input. The system will categorize cars into clusters (e.g., budget, luxury, etc.) and offer tailored recommendations that meet the user's needs, thus reducing the time and effort spent on car selection.

---

## Dataset

The system uses a dataset that includes various features of cars, such as:

- **Car Make and Model**
- **Price**
- **Fuel Type (Petrol, Diesel, etc.)**
- **Transmission Type (Automatic, Manual)**
- **Mileage**
- **Year of Manufacture**
- **Car Features (e.g., AC, Power Steering)**

The dataset was sourced from **CarDekho** and is cleaned and pre-processed to remove any inconsistencies. The dataset contains detailed car information that is essential for making accurate recommendations.

You can access the dataset here: [Car Details from CarDekho Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download).

---

## Features

- **Clustering**: Cars are grouped using the **KMeans** clustering algorithm into categories like 'Budget-Friendly', 'Mid-Range', and 'Luxury', making it easy for users to explore similar cars based on their preferences.
- **Price-Based Recommendations**: Users can input their budget, and the system will suggest cars that fall within their price range.
- **User Preferences**: Users can specify preferences such as fuel type and transmission type, and the system will filter the recommendations accordingly.
- **Data Visualizations**: Visualizations such as price distribution, fuel type breakdown, and car category segmentation provide valuable insights into the dataset.

---

## Demo

The Car Recommendation System can be run directly in **Google Colab**. Below is the link to the live demo:

[Try the Car Recommendation System on Google Colab](https://colab.research.google.com/drive/1BO0CpQkQ6QhGLHDbZwuP2_fHn5tDisCe?usp=sharing)

---

## Setup

### Requirements

Ensure you have the following Python libraries installed:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib**: For plotting visualizations.
- **seaborn**: For statistical data visualization.
- **scikit-learn**: For implementing KMeans clustering.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SudharsanamRK/car-recommendation-system.git

2. **Navigate to the project directory:**:
   ```bash
   cd car-recommendation-system
3. **Install required dependencies:** You can install all necessary libraries by running:
   ```bash   
   pip install -r requirements.txt
4. **Run the Program:** Ensure the dataset car-details-from-car-dekho.csv is in your project directory, and run the Python script to start the car recommendation process:

   ```bash
   python car_recommendation.py

---

## Code Walkthrough:
1. Data Preprocessing: The dataset is cleaned, and categorical variables like fuel type and transmission are encoded for the machine learning algorithm.

2. KMeans Clustering: The KMeans algorithm is applied to the data to create clusters that group similar cars together.

3. Recommendation Engine: Based on user input (budget, fuel type, transmission), the system filters and suggests cars from the relevant clusters.

4. Visualizations: We use matplotlib and seaborn to create visual representations of the data, helping users understand car price distribution, fuel types, and categories.

---

## Known Issues / Future Improvements

- The current recommendation system does not factor in user reviews or ratings.
- The clustering might not always provide optimal results for all car types and segments.
- Future versions could integrate more detailed features (e.g., user reviews, advanced filtering options).
- Add more diverse datasets for improved accuracy and broader recommendations.

---

## Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv('car-details-from-car-dekho.csv')
print(dataset.head())
print(dataset.tail())

# Print columns to confirm structure
print("Columns in dataset:", dataset.columns)

# Encode categorical variables
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
for column in categorical_columns:
    dataset[column] = pd.factorize(dataset[column])[0]

# Define features and scale them
X = dataset[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering to segment cars into price categories
kmeans = KMeans(n_clusters=3, random_state=42)
dataset['cluster'] = kmeans.fit_predict(X_scaled)

# Map cluster labels to car categories
dataset['category'] = dataset['cluster'].map({
    0: 'Budget-Friendly',
    1: 'Mid-Range',
    2: 'Luxury'
})

# Plot distribution of car prices by category
plt.figure(figsize=(10, 6))
sns.histplot(data=dataset, x='selling_price', hue='category', kde=True, palette='viridis')
plt.title("Car Price Distribution by Category")
plt.xlabel("Selling Price")
plt.ylabel("Number of Cars")
plt.legend(title='Category')
plt.show()

# Plot fuel type distribution for each category
plt.figure(figsize=(10, 6))
sns.countplot(data=dataset, x='fuel', hue='category', palette='viridis')
plt.title("Fuel Type Distribution by Car Category")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.legend(title='Category')
plt.show()

# Function to recommend cars based on budget and other preferences
def recommend_cars(budget, fuel_type=None, transmission=None):
    recommendations = dataset[(dataset['selling_price'] <= budget)]
    if fuel_type is not None:
        recommendations = recommendations[recommendations['fuel'] == fuel_type]
    if transmission is not None:
        recommendations = recommendations[recommendations['transmission'] == transmission]
    return recommendations[['year', 'km_driven', 'selling_price', 'category']]

# Example recommendations
print(recommend_cars(budget=500000, fuel_type=1, transmission=1).head())
```
