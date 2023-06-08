import numpy as np
import matplotlib.pyplot as plt

bids = np.linspace(0.03, 3, 100)
prices = np.array([200, 250, 300, 350, 400])
cost = 150
seed = 42
features = ['00', '01', '10', '11']

pricing_probabilities = {
    1: np.array([0.18, 0.22, 0.35, 0.20, 0.10]), # middle class 00-01
    2: np.array([0.05, 0.30, 0.20, 0.15, 0.25]), # mid-high class 10
    3: np.array([0.15, 0.13, 0.11, 0.19, 0.28]), # high class 11
}

class_from_feature = {
    '00': 1,
    '01': 1,
    '10': 2,
    '11': 3
}
features_from_class = {
    1: ['00', '01'],
    2: ['10'],
    3: ['11']
}

# plt.figure(0)
# plt.title("pricing_probabilities")
# plt.legend(["middle class", "mid-high class", "high class"])
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities[1], 'r')
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities[2], 'b')
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities[3], 'g')
# plt.show()

n_clicks_per_bid_by_class = {
    1: lambda x:np.sum(n_clicks_per_bid_by_feature[feature](x) for feature in features_from_class[1]),
    2: lambda x:np.sum(n_clicks_per_bid_by_feature[feature](x) for feature in features_from_class[2]),
    3: lambda x:np.sum(n_clicks_per_bid_by_feature[feature](x) for feature in features_from_class[3]),
}
n_clicks_per_bid_by_feature = {
    '00': lambda x:0.5*((1-np.exp(-x*1.5))*100+10),
    '01': lambda x:0.5*((1-np.exp(-x*1.5))*100+10),
    '10': lambda x:(1-np.exp(-x*2))*50+7,
    '11': lambda x:(1-np.exp(-x*2.5))*30+5,
}
# plt.figure(1)
# plt.title("n_clicks_per_bid_functions")
# plt.legend(["middle class", "mid-high class", "high class"])
# plt.plot(bids, n_clicks_per_bid_functions[1](bids), 'r')
# plt.plot(bids, n_clicks_per_bid_functions[2](bids), 'b')
# plt.plot(bids, n_clicks_per_bid_functions[3](bids), 'g')
# plt.show()
n_clicks_per_bid_sigma = 3.0

cum_cost_per_bid_by_class = {
    1: lambda x:np.sum(cum_cost_per_bid_by_feature[feature](x) for feature in features_from_class[1]),
    2: lambda x:np.sum(cum_cost_per_bid_by_feature[feature](x) for feature in features_from_class[2]),
    3: lambda x:np.sum(cum_cost_per_bid_by_feature[feature](x) for feature in features_from_class[3]),
}
cum_cost_per_bid_by_feature = {
    '00': lambda x:0.5*((1-np.exp(-x*1.5))*300+30),
    '01': lambda x:0.5*((1-np.exp(-x*1.5))*300+30),
    '10': lambda x:(1-np.exp(-x*2))*150+21,
    '11': lambda x:(1-np.exp(-x*2.5))*90+15,
}
# plt.figure(1)
# plt.title("cumulative_cost_per_bid_functions")
# plt.legend(["middle class", "mid-high class", "high class"])
# plt.plot(bids, cumulative_cost_per_bid_functions[1](bids), 'r')
# plt.plot(bids, cumulative_cost_per_bid_functions[2](bids), 'b')
# plt.plot(bids, cumulative_cost_per_bid_functions[3](bids), 'g')
# plt.show()
cum_cost_per_bid_sigma = 10.0
