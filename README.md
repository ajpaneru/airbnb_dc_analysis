# What Factors Influence Airbnb Prices in Washington, DC?

This project is for my Applied Data Science class.

The goal is to analyze Airbnb listings in Washington, DC and explore which factors
(neighbourhood, bedrooms, room type, amenities, etc.) are associated with higher nightly prices.

## Dataset

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/)

Steps:
1. Go to the link above in your browser.
2. Find **Washington, DC**.
3. Download the **"listings.csv"** (detailed listings data).
4. Save it into the `data/` folder as:

\`\`\`text
data/listings.csv
\`\`\`

On macOS, if it downloads to your `Downloads` folder, you can move it with:

\`\`\`bash
mv ~/Downloads/listings.csv data/listings.csv
\`\`\`

## Project Structure

\`\`\`text
airbnb_dc_analysis/
├─ data/
│  └─ listings.csv          # raw Airbnb data for DC
├─ figures/                 # generated plots
├─ src/
│  └─ airbnb_dc_analysis.py # main analysis script
├─ README.md
└─ requirements.txt
\`\`\`

## How to Run

1. (Optional) Create a virtual environment:

\`\`\`bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
\`\`\`

2. Install dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Make sure \`data/listings.csv\` exists.

4. Run the analysis:

\`\`\`bash
python src/airbnb_dc_analysis.py
\`\`\`

This will:
- Load and clean the Airbnb data
- Create basic features (like \`amenities_count\`)
- Generate exploratory plots and save them in \`figures/\`:
  - \`price_distribution.png\`
  - \`price_by_neighbourhood.png\`
  - \`bedrooms_vs_price.png\`
  - \`correlation_heatmap.png\`

## Outputs

You can use the generated plots directly in your class presentation:

- Distribution of nightly prices in DC
- Price differences across neighbourhoods
- Relationship between bedrooms and price
- Correlation heatmap of key numeric features

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
