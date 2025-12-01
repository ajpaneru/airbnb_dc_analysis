What Factors Influence Airbnb Prices in Washington, DC?
A Data Analysis + Machine Learning Project for Applied Data Science
📌 Overview
This project analyzes Airbnb listings in Washington, DC to understand which factors have the strongest impact on nightly prices. Using real-world datasets, exploratory analysis, feature engineering, and a Random Forest regression model, the project identifies key price drivers such as neighborhood, accommodates, amenities, and room type.
The goal of this project is to apply the full data science workflow — from data collection to modeling — and develop insights that are useful for travelers, hosts, and urban analysts.
📊 Key Questions
Which neighborhoods in DC have the highest and lowest Airbnb prices?
Do features like accommodates, room type, and amenities significantly impact nightly cost?
Does nearby crime activity correlate with Airbnb pricing?
Which features are most important in predicting price using machine learning?
📂 Dataset Sources
1. InsideAirbnb — Washington, DC
Source: https://insideairbnb.com/get-the-data/
Download the file named “listings.csv” under Washington, DC.
Place it here:
data/listings.csv
If downloaded to your macOS Downloads folder:
mv ~/Downloads/listings.csv data/listings.csv
2. DC Crime API (Real-Time Data)
Crime data for the past 30 days was collected using:
https://opendata.dc.gov/pages/opendata-api
This adds neighborhood-level context when analyzing price trends.
🏗️ Project Structure
airbnb_dc_analysis/
├── data/
│   └── listings.csv                # Raw Airbnb data
|   └── crime_last30.csv                # Raw crime data
├── figures/                        # Generated plots
│   ├── price_distribution.png
│   ├── price_by_neighbourhood.png
│   ├── airbnb_correlation_heatmap.png
│   ├── model_feature_importances.png
│   └── crime_by_neighbourhood_cluster.png
├── src/
│   └── airbnb_dc_analysis.py       # Main analysis & ML script
├── requirements.txt
└── README.md
⚙️ How to Run the Project
1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate          # Windows
2. Install dependencies
pip install -r requirements.txt
3. Ensure the dataset exists
data/listings.csv
4. Run the analysis
python src/airbnb_dc_analysis.py
This script will automatically clean data, engineer features, run the machine learning model, and generate charts.
📈 Generated Outputs
The script will generate figures in the figures/ folder:
Pricing Insights
price_distribution.png
Distribution of DC Airbnb prices (right-skewed with a long tail)
price_by_neighbourhood.png
Neighborhood-level median price map (Downtown, Navy Yard, Georgetown highly priced)
Feature Relationships
airbnb_correlation_heatmap.png
Shows which numeric features correlate with price
model_feature_importances.png
From the Random Forest model — “accommodates” and “room type” dominate
Crime & Context
crime_by_neighbourhood_cluster.png
Neighborhood crime clustering to explore contextual patterns
🤖 Machine Learning Model
A Random Forest Regressor was used to estimate Airbnb prices based on listing attributes.
🔑 Top Predictive Features:
accommodates
room_type
amenities_count
availability_365
review_scores_rating
The model helps quantify which features truly matter in price prediction.
🧼 Data Cleaning & Feature Engineering
Key processing steps include:
Removed price symbols and converted to numeric
Handled missing/invalid values
Removed extreme price outliers
Engineered amenities_count
Joined DC Crime API results to neighborhoods
Normalized categorical values
Prepared features for modeling
🧩 Insights & Conclusions
Neighborhood is one of the strongest drivers of Airbnb price.
Accommodates and room type heavily influence nightly cost.
Amenities significantly increase price (especially entire homes).
Crime data provides additional insight — areas with higher crime tend to have lower Airbnb prices.
Price distribution is heavily skewed, with most listings under $200 per night.
Machine learning confirms the relationships seen in exploratory analysis.
🧰 Tech Stack
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-Learn
API Requests
GitHub
🎓 About This Project
Created by Rojan Paneru (@03042926)
For Applied Data Science — Final Project
Fall 2025
