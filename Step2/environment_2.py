import parameters as param
import numpy as np

class Environment:
    def __init__(self, class_id):
        self.probabilities = param.pricing_probabilities[class_id]
        self.optimal = np.max(self.probabilities)
        self.n_arms = len(param.bids)

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
