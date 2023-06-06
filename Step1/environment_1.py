import parameters as param
import numpy as np


class Environment:
    def __init__(self, class_id):
        self.class_id = class_id
        self.probabilities = param.pricing_probabilities[class_id]
        self.optimal_arm = np.argmax(self.probabilities)
        self.max_bid = np.argmax(param.n_clicks_per_bid_functions[self.class_id](param.bids))
        self.n_clicks = round(param.n_clicks_per_bid_functions[self.class_id](param.bids)[self.max_bid])
        self.cumcost = param.cumulative_cost_per_bid_functions[self.class_id](self.max_bid)
        self.optimal = np.max(self.probabilities) * self.n_clicks * (param.prices[self.optimal_arm] - param.cost) - self.cumcost
        self.n_arms = len(self.probabilities)

    def round(self, pulled_arm):
        result = np.random.binomial(1, self.probabilities[pulled_arm], self.n_clicks)
        reward = np.sum(result) * (param.prices[pulled_arm] - param.cost) - self.cumcost
        return np.sum(result), self.n_clicks-np.sum(result), reward
