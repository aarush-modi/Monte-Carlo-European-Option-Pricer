import numpy as np
import matplotlib.pyplot as plt

#Parameters
S0 = float(input("What is the initial stock price? "))
K = float(input("What is the strike price? "))
t = float(input("What is the time to maturity? "))
r = float(input("What is the risk-free rate as a decimal? "))
sigma = float(input("What is the volitality of the option as a decimal? "))
sim = 1000000 #Number of simulations

print("Computing")

#Generate random samples of stock prices at maturity
Z = np.random.standard_normal(sim)
ST = S0 * np.exp((r - 0.5 * sigma * sigma) * t + sigma * np.sqrt(t) * Z)

#Calculate the payoff for each path
payoffs = np.maximum(ST - K, 0)

#Discount the average payoff to present value
option_price = np.exp(-r * t) * np.mean(payoffs)

#Print resluts
print(f"The estimated option price using Monte Carlo is: {option_price:.2f}")

#Histogram plotting
plt.hist(payoffs, bins=50, alpha=0.75)
plt.title('Histogram of Payoffs')
plt.xlabel('Payoff')
plt.ylabel('Frequency')
plt.show()