"""Original file is located at
    https://colab.research.google.com/drive/1qzD33jL5Ts9cMq90B7S09SMmndD4D2nn
"""

!pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

def train_regression_model(generation_path, weather_path):
    """
    Loads and merges data, then trains a regression model to predict power from weather.
    It calculates a dynamic threshold for anomaly detection based on prediction errors.

    Returns:
        (GradientBoostingRegressor, float, pd.DataFrame): A tuple containing the trained model,
        the anomaly threshold, and the training data for visualization.
    """
    # --- 1. Load and Merge Data ---
    print("Loading power generation and weather data...")
    try:
        df_gen = pd.read_csv(generation_path)
        df_weather = pd.read_csv(weather_path)
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Please ensure both CSV files are available.")
        return None, None, None

    # Convert DATE_TIME columns to datetime objects
    df_gen['DATE_TIME'] = pd.to_datetime(df_gen['DATE_TIME'], format='%d-%m-%Y %H:%M')
    df_weather['DATE_TIME'] = pd.to_datetime(df_weather['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # Merge the two dataframes on the nearest timestamp
    df_merged = pd.merge_asof(df_gen.sort_values('DATE_TIME'), df_weather.sort_values('DATE_TIME'), on='DATE_TIME')
    df_merged.dropna(inplace=True)

    # --- 2. Train Weather-Based Prediction Model ---
    print("Training a regression model to learn the relationship between weather and power...")

    weather_features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    target = 'AC_POWER'

    # We only want to learn from daytime data where generation is expected
    training_data = df_merged[df_merged['IRRADIATION'] > 0.1].copy()

    # Initialize and train the Gradient Boosting Regressor
    regression_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    regression_model.fit(training_data[weather_features], training_data[target])

    # --- 3. Establish Anomaly Threshold ---
    print("Calculating anomaly threshold based on prediction errors...")

    # Predict on the training data to find normal error levels
    predictions = regression_model.predict(training_data[weather_features])
    # Calculate the error (residuals)
    training_data['residuals'] = training_data[target] - predictions

    # The threshold is 3 standard deviations from the mean error
    residual_std = training_data['residuals'].std()
    threshold = residual_std * 3

    print(f"Anomaly threshold established: Any deviation greater than {threshold:.2f} kW will be flagged.")

    return regression_model, threshold, training_data

def is_anomaly_with_regression(ac_power, irradiation, ambient_temp, module_temp, model, threshold):
    """
    Checks if a single data point is an anomaly using a regression model and a fixed threshold.

    Returns:
        bool: True if the point is an anomaly, False otherwise.
    """
    # Define the exact feature order the model was trained on
    weather_features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']

    # Create a DataFrame for the single point
    data = {
        'IRRADIATION': [irradiation],
        'AMBIENT_TEMPERATURE': [ambient_temp],
        'MODULE_TEMPERATURE': [module_temp]
    }
    point_df = pd.DataFrame(data)

    # Ensure the column order matches the training order before predicting
    expected_power = model.predict(point_df[weather_features])[0]

    # Calculate the absolute error (residual)
    absolute_error = abs(ac_power - expected_power)

    # Compare the error to our threshold
    return absolute_error > threshold

def check_files_for_anomalies(generation_filepath, weather_filepath, model, threshold):
    """
    Reads, validates, and merges generation and weather CSV files,
    then checks each row for anomalies and visualizes the results.
    """
    print(f"\n--- Checking files: {generation_filepath} and {weather_filepath} ---")
    try:
        df_gen_check = pd.read_csv(generation_filepath)
        df_weather_check = pd.read_csv(weather_filepath)

        # --- Validation ---
        if df_gen_check.empty or df_weather_check.empty:
            print("Error: One or both of the provided CSV files are empty.")
            return

        gen_cols = ['DATE_TIME', 'AC_POWER']
        weather_cols = ['DATE_TIME', 'IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE']

        if not all(col in df_gen_check.columns for col in gen_cols):
            print(f"Error: The generation CSV file must contain the columns: {gen_cols}")
            return
        if not all(col in df_weather_check.columns for col in weather_cols):
            print(f"Error: The weather CSV file must contain the columns: {weather_cols}")
            return

        # --- Preprocessing and Merging ---
        df_gen_check['DATE_TIME'] = pd.to_datetime(df_gen_check['DATE_TIME'])
        df_weather_check['DATE_TIME'] = pd.to_datetime(df_weather_check['DATE_TIME'])
        df_to_check = pd.merge_asof(df_gen_check.sort_values('DATE_TIME'), df_weather_check.sort_values('DATE_TIME'), on='DATE_TIME')
        df_to_check.dropna(inplace=True)

        if df_to_check.empty:
            print("Error: No matching timestamps found between the two files.")
            return

        # --- Anomaly Checking ---
        df_to_check['is_anomaly'] = df_to_check.apply(
            lambda row: is_anomaly_with_regression(
                ac_power=row['AC_POWER'],
                irradiation=row['IRRADIATION'],
                ambient_temp=row['AMBIENT_TEMPERATURE'],
                module_temp=row['MODULE_TEMPERATURE'],
                model=model,
                threshold=threshold
            ),
            axis=1
        )

        anomalies_df = df_to_check[df_to_check['is_anomaly']]

        # --- Visualization of Results ---
        print("Generating results plot...")
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))

        # Plot the entire power series from the user's file
        plt.plot(df_to_check['DATE_TIME'], df_to_check['AC_POWER'],
                 color='dodgerblue', label='AC Power from File')

        # Highlight the anomalies
        if not anomalies_df.empty:
            plt.scatter(anomalies_df['DATE_TIME'], anomalies_df['AC_POWER'],
                        color='red', s=80, label='Anomaly Detected', zorder=5)
            print(f"\nDetected {len(anomalies_df)} anomalies.")
        else:
            print("\nNo anomalies were detected.")

        plt.title(f'Anomaly Check Results for {generation_filepath.split("/")[-1]}', fontsize=16)
        plt.xlabel('Date and Time', fontsize=12)
        plt.ylabel('AC Power (kW)', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: One of the files was not found. Please check the paths and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main Execution Block ---
if __name__ == '__main__':
    print("--- Initializing Regression-Based Anomaly Detection System ---")

    # Train our model and get the dynamic threshold
    regression_model, anomaly_threshold, training_data_for_viz = train_regression_model(
        generation_path='Plant_1_Generation_Data.csv',
        weather_path='Plant_1_Weather_Sensor_Data.csv'
    )

    if regression_model:
        # The initial demonstration with hardcoded cases is now omitted to focus on the file checker.
        # --- Interactive File Check Loop ---
        while True:
            print("\n" + "="*50)
            gen_filepath = input("Enter the path to the POWER GENERATION CSV file (or type 'exit' to quit): ")
            if gen_filepath.lower() == 'exit':
                break

            weather_filepath = input("Enter the path to the corresponding WEATHER DATA CSV file: ")

            check_files_for_anomalies(gen_filepath, weather_filepath, regression_model, anomaly_threshold)

    print("\n--- Process Complete ---")
