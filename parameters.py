import numpy as np
import matplotlib.pyplot as plt

bids = np.linspace(0.03, 3, 100)
prices = np.array([200, 250, 300, 350, 400])
cost = 150
seed = 42
feature_combos = ['00', '01', '10', '11']
features_names = ["F1", "F2"]
confidence = 0.95

pricing_probabilities = {
    1: np.array([0.05, 0.35, 0.15, 0.10, 0.13]),  # middle class 00-01
    2: np.array([0.18, 0.22, 0.35, 0.20, 0.10]),  # mid-high class 10
    3: np.array([0.15, 0.13, 0.11, 0.19, 0.26]),  # high class 11
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
# plt.plot(np.linspace(200, 400, 5), pricing_probabilities[1], 'r')
# plt.plot(np.linspace(200, 400, 5), pricing_probabilities[2], 'b')
# plt.plot(np.linspace(200, 400, 5), pricing_probabilities[3], 'g')
# labels=[200,250,300,350,400]
# plt.xticks(np.linspace(200, 400, 5), labels)
# plt.legend(["middle class", "mid-high class", "high class"])
# plt.show()

n_clicks_per_bid_by_class = {
    1: lambda x: np.sum(n_clicks_per_bid_by_feature[feature](x) for feature in features_from_class[1]),
    2: lambda x: np.sum(n_clicks_per_bid_by_feature[feature](x) for feature in features_from_class[2]),
    3: lambda x: np.sum(n_clicks_per_bid_by_feature[feature](x) for feature in features_from_class[3]),
}
n_clicks_per_bid_by_feature = {
    '00': lambda x: 0.5 * ((1 - np.exp(-x * 1.5)) * 100 + 10),
    '01': lambda x: 0.5 * ((1 - np.exp(-x * 1.5)) * 100 + 10),
    '10': lambda x: (1 - np.exp(-x * 2)) * 50 + 7,
    '11': lambda x: (1 - np.exp(-x * 2.5)) * 30 + 5,
}
# plt.figure(1)
# plt.title("n_clicks_per_bid_functions")
# plt.plot(bids, n_clicks_per_bid_by_class[1](bids), 'r')
# plt.plot(bids, n_clicks_per_bid_by_class[2](bids), 'b')
# plt.plot(bids, n_clicks_per_bid_by_class[3](bids), 'g')
# plt.legend(["middle class", "mid-high class", "high class"])
# plt.show()
n_clicks_per_bid_sigma = 3.0

cum_cost_per_bid_by_class = {
    1: lambda x: np.sum(cum_cost_per_bid_by_feature[feature](x) for feature in features_from_class[1]),
    2: lambda x: np.sum(cum_cost_per_bid_by_feature[feature](x) for feature in features_from_class[2]),
    3: lambda x: np.sum(cum_cost_per_bid_by_feature[feature](x) for feature in features_from_class[3]),
}
cum_cost_per_bid_by_feature = {
    '00': lambda x: 0.5 * ((1 - np.exp(-x * 1.5)) * 300 + 30),
    '01': lambda x: 0.5 * ((1 - np.exp(-x * 1.5)) * 300 + 30),
    '10': lambda x: (1 - np.exp(-x * 2)) * 150 + 21,
    '11': lambda x: (1 - np.exp(-x * 2.5)) * 90 + 15,
}
# plt.figure(1)
# plt.title("cumulative_cost_per_bid_functions")
# plt.plot(bids, cum_cost_per_bid_by_class[1](bids), 'r')
# plt.plot(bids, cum_cost_per_bid_by_class[2](bids), 'b')
# plt.plot(bids, cum_cost_per_bid_by_class[3](bids), 'g')
# plt.legend(["middle class", "mid-high class", "high class"])
# plt.show()
cum_cost_per_bid_sigma = 10.0

pricing_probabilities_by_phase = {
    1: np.array([0.05, 0.35, 0.15, 0.10, 0.13]),  # first phase
    # price*prob [2.5, 35, 22.5, 20, 32.5]
    2: np.array([0.18, 0.20, 0.38, 0.22, 0.15]),  # second phase
    # price*prob [9, 20, 57, 44, 37.5]
    3: np.array([0.28, 0.16, 0.12, 0.10, 0.06]),  # third phase
    # price*prob [14, 16, 18, 20, 15]
}

# plt.figure(0)
# plt.title("pricing_probabilities step 5")
# plt.legend(["normal period", "holiday period", "sales period"])
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities_by_phase[1], 'r')
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities_by_phase[2], 'b')
# plt.plot(np.linspace(0, 5, 5), pricing_probabilities_by_phase[3], 'g')
# plt.show()

pricing_probabilities_by_phase_6b = {
    1: np.array([0.05, 0.35, 0.15, 0.10, 0.13]),  # first phase
    # price*prob [ 2.5, 35. , 22.5, 20. , 32.5]
    2: np.array([0.45, 0.15, 0.65, 0.50, 0.26]),  # second phase
    # price*prob [ 22.5,  15. ,  97.5, 100. ,  65. ]
    3: np.array([0.28, 0.16, 0.12, 0.10, 0.06]),  # third phase
    # price*prob [14., 16., 18., 20., 15.]
    4: np.array([0.85, 0.35, 0.25, 0.05, 0.10]),  # fourth phase
    # price*prob [42.5, 35. , 37.5, 10. , 25. ]
    5: np.array([0.30, 0.05, 0.15, 0.10, 0.40]),  # fifth phase
    # price*prob [ 15. ,   5. ,  22.5,  20. , 100. ]
}

#EXP3 bid fixed so num of click fixed
step6_fixed_bid = 2.79
