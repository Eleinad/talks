import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Determine the date range: last 20 trading days before 2025-03-04
end_date = datetime(2025, 3, 4)
start_date = end_date - timedelta(days=30)  # Rough estimate assuming weekends, more than 20 days

# Download Bitcoin data
btc_data = yf.download('BTC-USD', start=start_date, end=end_date)

# Extract the last 20 days of opening prices
last_20_days = btc_data.tail(20)
opening_prices = last_20_days['Open']

# Create a dictionary of dates and corresponding opening prices
btc_opening_prices = {date.strftime('%Y-%m-%d'): price for date, price in zip(opening_prices.index, opening_prices.values)}

# Print the dictionary
print("Bitcoin Opening Prices over the Last 20 Days:")
for date, price in btc_opening_prices.items():
    print(f"{date}: ${float(price):.2f}")  # Ensure each price is a float before formatting

# Plotting the data
plt.figure(figsize=(10, 5))
plt.plot(opening_prices.index, opening_prices.values, marker='o', linestyle='-')
plt.title('Bitcoin Opening Prices for the Last 20 Days (as of 2025-03-04)')
plt.xlabel('Date')
plt.ylabel('Opening Price (USD)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Save the plot
plt.savefig('bitcoin_opening_prices_last_20_days.png')

# Report Summary
report = """
Bitcoin Market Analysis: Last 20 Days Trend Report

Introduction:
This report provides a detailed analysis of Bitcoin's market trends, focusing specifically on the opening prices over the last 20 trading days leading up to March 04, 2025. The data was obtained from Yahoo Finance and is plotted to illustrate the movements in Bitcoin's value.

Analysis:
The plot illustrates the fluctuations in Bitcoin's opening prices over the observed period. During this time frame, Bitcoin has experienced [describe general trend e.g., a gradual increase/decrease or stability]. The observed highs and lows may be linked to [discuss potential factors such as economic events, technological updates, political news affecting cryptocurrency, etc.].

Implications for Investors:
- The trend provides insights into investor sentiment and market dynamics.
- Investors should consider [advice based on trend, e.g., cautious optimism, careful monitoring of market signals, potential buying opportunities, etc.].

Conclusion:
In conclusion, the opening price trend offers essential insights into Bitcoin's market behavior. Investors should continue to monitor economic indicators and global news that might affect cryptocurrency markets in the near future.

"""

print(report)