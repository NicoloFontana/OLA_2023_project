import numpy as np
import matplotlib.pyplot as plt

pricing_probabilities = {
    1: np.array([0.18, 0.22, 0.35, 0.20, 0.10]), # middle class
    2: np.array([0.05, 0.30, 0.20, 0.15, 0.23]), # mid-high class
    3: np.array([0.15, 0.13, 0.11, 0.19, 0.28]), # high class
}
plt.figure(0)
plt.title("pricing_probabilities")
plt.legend(["middle class", "mid-high class", "high class"])
plt.plot(np.linspace(0, 5, 5), pricing_probabilities[1], 'r')
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities[2], 'b')
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities[3], 'g')
plt.show()


bids = np.linspace(0.03, 3, 100)

n_clicks_per_bid_functions = {
    1: lambda x:(1-np.exp(-x*1.5))*100+10,
    2: lambda x:(1-np.exp(-x*2))*50+7,
    3: lambda x:(1-np.exp(-x*2.5))*30+5,
}
plt.figure(1)
plt.title("n_clicks_per_bid_functions")
plt.legend(["middle class", "mid-high class", "high class"])
plt.plot(bids, n_clicks_per_bid_functions[1](bids), 'r')
# plt.plot(bids, n_clicks_per_bid_functions[2](bids), 'b')
# plt.plot(bids, n_clicks_per_bid_functions[3](bids), 'g')
plt.show()
n_clicks_per_bid_sigma = 3.0

cumulative_cost_per_bid_functions = {
    1: lambda x:(1-np.exp(-x*1.5))*300+30,
    2: lambda x:(1-np.exp(-x*2))*150+21,
    3: lambda x:(1-np.exp(-x*2.5))*90+15,
}
plt.figure(1)
plt.title("cumulative_cost_per_bid_functions")
plt.legend(["middle class", "mid-high class", "high class"])
plt.plot(bids, cumulative_cost_per_bid_functions[1](bids), 'r')
# plt.plot(bids, cumulative_cost_per_bid_functions[2](bids), 'b')
# plt.plot(bids, cumulative_cost_per_bid_functions[3](bids), 'g')
plt.show()
cumulative_cost_per_bid_sigma = 10.0

prices = np.array([200, 250, 300, 350, 400])
cost = 150