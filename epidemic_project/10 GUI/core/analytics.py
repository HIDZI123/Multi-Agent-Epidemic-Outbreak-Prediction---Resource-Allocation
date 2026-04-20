from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

from core.policy import Prediction


class PredictiveAnalytics:
    def __init__(self, history_window: int = 20):
        self.history_window = history_window

    def predict_hospital_demand(self, historical_data: pd.DataFrame, horizon: int = 5) -> Prediction:
        if len(historical_data) < self.history_window:
            return Prediction("hospital_demand", 0, 0, horizon, 0.5)

        recent_data = historical_data["Total Hospital Occupancy"].tail(self.history_window).values
        x_vals = np.arange(len(recent_data)).reshape(-1, 1)
        model = LinearRegression().fit(x_vals, recent_data)
        future_x = np.arange(len(recent_data), len(recent_data) + horizon).reshape(-1, 1)
        predicted = float(model.predict(future_x)[-1])

        volatility = float(np.std(np.diff(recent_data))) if len(recent_data) > 1 else 0.0
        confidence = max(0.3, 1.0 - volatility / max(1.0, recent_data[-1]))
        return Prediction("hospital_demand", float(recent_data[-1]), predicted, horizon, confidence)

    def predict_infection_peak(self, historical_data: pd.DataFrame) -> Prediction:
        if len(historical_data) < self.history_window:
            return Prediction("infection_peak", 0, 0, 10, 0.5)

        infected_total = (historical_data["Infected (Untreated)"] + historical_data["Hospitalized"]).values
        if len(infected_total) < 11:
            current = float(infected_total[-1])
            return Prediction("infection_peak", current, current, 10, 0.4)

        smoothed = savgol_filter(infected_total, 11, 2)
        derivatives = np.diff(smoothed)
        trend = float(np.mean(derivatives[-5:])) if len(derivatives) >= 5 else 0.0
        current = float(infected_total[-1])

        if trend > 0:
            steps_to_peak = max(5, int(current / max(0.1, trend)))
            predicted_peak = current + trend * steps_to_peak
        else:
            steps_to_peak = 0
            predicted_peak = current

        confidence = 0.7 if abs(trend) > 1 else 0.4
        return Prediction("infection_peak", current, float(predicted_peak), steps_to_peak, confidence)
