import numpy as np
import matplotlib.pyplot as plt

#Parameters
S0 = float(input("What is the initial stock price? "))
K = float(input("What is the strike price? "))
t = float(input("What is the time to maturity? "))
r = float(input("What is the risk-free rate as a decimal? "))
sigma = float(input("What is the volitality of the option as a decimal? "))

def monte_carlo_price(S0, K, T, r, sigma, sim = 100000):
    #Generate random samples of stock prices at maturity
    Z = np.random.standard_normal(sim)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    #Calculate the payoff for each path
    payoff = np.maximum(ST - K, 0)
    #Discount the average payoff to present value
    price = np.exp(-r * T) * np.mean(payoff)
    stderr = np.std(payoff) / np.sqrt(sim)
    return price, stderr




#Sensitivity to volatility
vols = np.linspace(0.1, 0.5, 20)
prices = []
errors = []

for sigma in vols:
    price, err = monte_carlo_price(S0, K, t, r, sigma)
    prices.append(price)
    errors.append(err)

fig = plt.figure(figsize=(8, 5))
fig.canvas.manager.set_window_title("Volatility Sensitivity")
plt.errorbar(vols, prices, yerr=1.96*np.array(errors), fmt='-o', capsize=4)
plt.xlabel("Volatility (σ)")
plt.ylabel("Call Option Price")
plt.title("European Call Price vs Volatility")
plt.grid(True)
plt.tight_layout()
plt.show(block = False)

#Heatmap of Option Price vs Volatility & Stock Price
S0_range = np.linspace(80, 120, 20)
vol_range = np.linspace(0.1, 0.5, 20)
price_matrix = np.zeros((len(S0_range), len(vol_range)))

for i, S0_val in enumerate(S0_range):
    for j, sigma_val in enumerate(vol_range):
        price, _ = monte_carlo_price(S0_val, K, t, r, sigma_val)
        price_matrix[i, j] = price

fig = plt.figure(figsize=(8, 6))
fig.canvas.manager.set_window_title("Option Price Heatmap")
plt.imshow(price_matrix, extent=[vol_range[0], vol_range[-1], S0_range[0], S0_range[-1]],
           origin='lower', aspect='auto', cmap='viridis')

plt.colorbar(label='Option Price')
plt.xlabel('Volatility (σ)')
plt.ylabel('Stock Price (S₀)')
plt.title('Option Price Sensitivity Heatmap')
plt.tight_layout()
plt.show()
