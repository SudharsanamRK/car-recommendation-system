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

You can access the dataset here: [Car Details from CarDekho Dataset]([https://github.com/SudharsanamRK/car-recommendation-system/blob/main/car-details-from-car-dekho.csv](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download)).

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
