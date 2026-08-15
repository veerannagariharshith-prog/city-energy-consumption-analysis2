# City Energy Consumption Analysis & Prediction System

A complete Python project that generates synthetic electricity-consumption data for five city zones, analyzes usage patterns, creates visualizations, and predicts next-day energy consumption.

## Project structure

```text
city_energy_consumption/
├── city_energy_analysis.py
├── City_Energy_Consumption_Analysis.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── city_energy.csv
└── outputs/
    ├── monthly_energy_trend.png
    ├── correlation_heatmap.png
    └── event_vs_non_event.png
```

## Dataset

The project generates **365 days × 5 zones = 1,825 readings**.

Features:

- `Date` – daily reading date
- `ZoneID` – city zone identifier (1–5)
- `AvgTemperature` – average daily temperature in °C
- `Humidity` – daily humidity percentage
- `SpecialEvent` – 0/1 event indicator
- `EnergyConsumption` – daily energy usage in kWh

The synthetic generator includes seasonality, weather effects, event effects, weekend effects, zone-specific baselines, and random noise.

## Setup

Recommended Python version: 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Python program

From the project directory:

```bash
python city_energy_analysis.py
```

The program will:

1. Generate the dataset.
2. Clean and validate it.
3. Save `data/city_energy.csv`.
4. Create three plots in `outputs/`.
5. Train a Random Forest next-day prediction model.
6. Report test Mean Absolute Error (MAE).
7. Start an interactive console for tomorrow's prediction.

Example inputs:

```text
Zone ID (1-5): 3
Tomorrow's temperature (°C): 31
Tomorrow's humidity (%): 68
Special event? (0 = No, 1 = Yes): 0
```

## Run the notebook

Start Jupyter:

```bash
jupyter notebook
```

Open:

`City_Energy_Consumption_Analysis.ipynb`

Run all cells from top to bottom.

## Analysis performed

### 1. Monthly trends

The notebook calculates average daily consumption by month and visualizes the overall trend.

### 2. Zone-level comparison

Average, minimum, and maximum consumption are calculated for each zone. The different baselines demonstrate that zones can have different demand profiles.

### 3. Correlation analysis

A correlation heatmap compares temperature, humidity, special events, and energy consumption. Because the data is synthetic, correlations should be interpreted as properties of the simulation rather than real-world measurements.

### 4. Event vs non-event usage

Average consumption is compared between event and non-event days. The generator intentionally gives special-event days an additional demand component, so event days should generally show higher usage.

## Prediction approach

The model predicts:

`TargetNextDayConsumption`

for the same zone.

Input features:

- Zone ID
- Average temperature
- Humidity
- Special-event indicator

A **Random Forest Regressor** is used. The split is time-aware: earlier dates are used for training and later dates for testing. This avoids randomly mixing future observations into the training set.

The primary evaluation metric is **Mean Absolute Error (MAE)**:

```text
MAE = mean(|actual - predicted|)
```

Lower MAE indicates better prediction accuracy.

## Error handling

The console interface checks:

- Zone ID must be 1–5.
- Temperature must be between -20 and 60 °C.
- Humidity must be between 0 and 100%.
- Event must be 0 or 1.
- Invalid/missing numeric values are caught and reported without crashing the program.

The data-cleaning function also handles invalid dates, numeric conversion failures, missing required fields, invalid humidity/event values, and non-positive consumption values.

## GitHub upload

Create a repository and upload the project files:

```bash
git init
git add .
git commit -m "Build city energy consumption analysis and prediction system"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Do not commit large generated files unnecessarily; keep the repository focused on the notebook, Python source, README, requirements, dataset, and useful plots.

## LinkedIn demo idea

Share:

- The monthly trend chart.
- The correlation heatmap.
- The event vs non-event chart.
- The model's MAE.
- A short explanation of how the console accepts tomorrow's weather/event inputs and returns predicted consumption.
- The GitHub repository URL.

## Important note

This dataset is synthetic and intended for learning/project demonstration. The model should not be treated as a real electricity-grid forecasting system without validated historical data, additional calendar features, lagged consumption features, and appropriate production monitoring.
