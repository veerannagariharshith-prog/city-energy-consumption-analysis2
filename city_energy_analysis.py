"""
City Energy Consumption Analysis & Prediction System

Generates a synthetic 1-year, 5-zone dataset, performs analysis/visualization,
trains a next-day consumption model, evaluates MAE, and provides a console UI.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


RANDOM_STATE = 42
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_FILE = DATA_DIR / "city_energy.csv"


def generate_dataset(days=365, zones=None, seed=RANDOM_STATE):
    """Generate realistic synthetic daily electricity readings."""
    rng = np.random.default_rng(seed)
    zones = zones or {
        1: {"base": 4200, "temp_sensitivity": 95, "event_effect": 650},
        2: {"base": 5100, "temp_sensitivity": 115, "event_effect": 800},
        3: {"base": 3600, "temp_sensitivity": 80, "event_effect": 500},
        4: {"base": 5900, "temp_sensitivity": 130, "event_effect": 900},
        5: {"base": 4650, "temp_sensitivity": 100, "event_effect": 700},
    }

    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []

    for zone_id, cfg in zones.items():
        phase = rng.uniform(0, 2 * np.pi)
        zone_offset = rng.normal(0, 100)

        for i, date in enumerate(dates):
            # Seasonal temperature plus daily noise.
            temp = (
                27
                + 7.5 * np.sin(2 * np.pi * (i - 105) / 365)
                + 1.8 * np.sin(2 * np.pi * i / 30 + phase)
                + rng.normal(0, 1.4)
            )
            temp = float(np.clip(temp, 16, 39))

            humidity = (
                70
                - 0.75 * (temp - 27)
                + 7 * np.sin(2 * np.pi * (i + 35) / 365)
                + rng.normal(0, 5)
            )
            humidity = float(np.clip(humidity, 35, 95))

            # Roughly 10% special-event days, with slightly more on weekends.
            event_prob = 0.08 + (0.04 if date.dayofweek >= 5 else 0)
            event = int(rng.random() < event_prob)

            # Cooling/heating demand rises with distance from a comfortable 24 C.
            weather_load = cfg["temp_sensitivity"] * max(temp - 24, 0)
            humidity_load = 8 * max(humidity - 65, 0)
            weekend_adjustment = -180 if date.dayofweek >= 5 else 0
            trend = 0.10 * i
            noise = rng.normal(0, 180)

            consumption = (
                cfg["base"]
                + zone_offset
                + weather_load
                + humidity_load
                + cfg["event_effect"] * event
                + weekend_adjustment
                + trend
                + noise
            )
            consumption = max(consumption, 1000)

            rows.append(
                {
                    "Date": date,
                    "ZoneID": zone_id,
                    "AvgTemperature": round(temp, 2),
                    "Humidity": round(humidity, 2),
                    "SpecialEvent": event,
                    "EnergyConsumption": round(consumption, 2),
                }
            )

    return pd.DataFrame(rows)


def clean_data(df):
    """Validate, clean, and standardize the dataset."""
    required = [
        "Date", "ZoneID", "AvgTemperature", "Humidity",
        "SpecialEvent", "EnergyConsumption"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    numeric_cols = [
        "ZoneID", "AvgTemperature", "Humidity",
        "SpecialEvent", "EnergyConsumption"
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=required).copy()
    out["ZoneID"] = out["ZoneID"].astype(int)
    out["SpecialEvent"] = out["SpecialEvent"].astype(int)

    out = out[
        out["Humidity"].between(0, 100)
        & out["SpecialEvent"].isin([0, 1])
        & (out["EnergyConsumption"] > 0)
    ].copy()

    out = out.sort_values(["ZoneID", "Date"]).reset_index(drop=True)
    return out


def add_next_day_target(df):
    """Create the target: next day's consumption for the same zone."""
    out = df.sort_values(["ZoneID", "Date"]).copy()
    out["TargetNextDayConsumption"] = (
        out.groupby("ZoneID")["EnergyConsumption"].shift(-1)
    )
    return out.dropna(subset=["TargetNextDayConsumption"]).copy()


def train_model(df):
    """Time-aware split and Random Forest training."""
    features = ["ZoneID", "AvgTemperature", "Humidity", "SpecialEvent"]
    data = add_next_day_target(df)

    # Use the earliest 80% of dates for training and the latest 20% for testing.
    unique_dates = np.sort(data["Date"].unique())
    split_idx = int(len(unique_dates) * 0.80)
    split_date = unique_dates[split_idx]

    train = data[data["Date"] < split_date]
    test = data[data["Date"] >= split_date]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[features], train["TargetNextDayConsumption"])
    predictions = model.predict(test[features])
    mae = mean_absolute_error(test["TargetNextDayConsumption"], predictions)

    return model, features, train, test, predictions, mae


def create_visualizations(df, output_dir=OUTPUT_DIR):
    """Create the three required plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    monthly = (
        df.assign(Month=df["Date"].dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)["EnergyConsumption"]
        .mean()
    )

    plt.figure(figsize=(11, 5))
    sns.lineplot(data=monthly, x="Month", y="EnergyConsumption", marker="o")
    plt.xticks(rotation=45)
    plt.title("Average Daily Energy Consumption by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Consumption (kWh)")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_energy_trend.png", dpi=160)
    plt.close()

    corr_cols = [
        "AvgTemperature", "Humidity", "SpecialEvent", "EnergyConsumption"
    ]
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        df[corr_cols].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close()

    event_means = (
        df.assign(EventStatus=df["SpecialEvent"].map({0: "No Event", 1: "Event"}))
        .groupby("EventStatus")["EnergyConsumption"]
        .mean()
        .reindex(["No Event", "Event"])
    )
    plt.figure(figsize=(7, 5))
    plt.bar(event_means.index, event_means.values)
    plt.title("Average Energy Consumption: Event vs Non-Event")
    plt.xlabel("Day Type")
    plt.ylabel("Average Consumption (kWh)")
    plt.tight_layout()
    plt.savefig(output_dir / "event_vs_non_event.png", dpi=160)
    plt.close()


def run_pipeline():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df = clean_data(df)
    df.to_csv(DATA_FILE, index=False)

    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly_zone = (
        df.groupby(["Month", "ZoneID"])["EnergyConsumption"]
        .mean()
        .reset_index()
    )
    zone_summary = (
        df.groupby("ZoneID")["EnergyConsumption"]
        .agg(["mean", "min", "max"])
        .round(2)
    )

    create_visualizations(df)
    model, features, train, test, predictions, mae = train_model(df)

    print(f"Dataset saved to: {DATA_FILE}")
    print(f"Rows: {len(df)}")
    print("\nAverage consumption by zone:")
    print(zone_summary)
    print(f"\nTest MAE: {mae:.2f} kWh")

    return df, model, features, mae, monthly_zone, zone_summary


def predict_tomorrow(model, zone_id, temperature, humidity, event):
    """Predict next-day consumption from console inputs."""
    if not isinstance(zone_id, int) or zone_id not in [1, 2, 3, 4, 5]:
        raise ValueError("Zone ID must be an integer from 1 to 5.")
    if not np.isfinite(temperature) or not -20 <= temperature <= 60:
        raise ValueError("Temperature must be between -20 and 60 °C.")
    if not np.isfinite(humidity) or not 0 <= humidity <= 100:
        raise ValueError("Humidity must be between 0 and 100%.")
    if event not in [0, 1]:
        raise ValueError("Event indicator must be 0 or 1.")

    x = pd.DataFrame([{
        "ZoneID": zone_id,
        "AvgTemperature": temperature,
        "Humidity": humidity,
        "SpecialEvent": event,
    }])
    return float(model.predict(x)[0])


def console_interface(model):
    """Interactive prediction console with input validation."""
    print("\n--- Next-Day Energy Prediction ---")
    try:
        zone_id = int(input("Zone ID (1-5): ").strip())
        temperature = float(input("Tomorrow's temperature (°C): ").strip())
        humidity = float(input("Tomorrow's humidity (%): ").strip())
        event = int(input("Special event? (0 = No, 1 = Yes): ").strip())

        prediction = predict_tomorrow(
            model, zone_id, temperature, humidity, event
        )
        print(f"\nPredicted next-day consumption: {prediction:,.2f} kWh")
    except ValueError as exc:
        print(f"\nInput error: {exc}")
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")


if __name__ == "__main__":
    df, model, features, mae, monthly_zone, zone_summary = run_pipeline()
    console_interface(model)
