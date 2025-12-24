import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load dataset
df = pd.read_csv("data/retail_sales.csv")

# Rename columns for Prophet
df.columns = ["ds", "y"]

# Convert date column to datetime
df["ds"] = pd.to_datetime(df["ds"])

# Plot historical sales
plt.figure(figsize=(10, 5))
plt.plot(df["ds"], df["y"], label="Historical Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Historical Sales Data")
plt.legend()
plt.show()

# Initialize and train model
model = Prophet()
model.fit(df)

# Create future dataframe (next 30 days)
future = model.make_future_dataframe(periods=30)

# Forecast
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.show()

# Plot trend and seasonality
model.plot_components(forecast)
plt.show()

# Save forecast results
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
    "data/sales_forecast.csv", index=False
)

print("Forecast generated and saved as sales_forecast.csv")
