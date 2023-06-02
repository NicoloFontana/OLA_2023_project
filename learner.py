import numpy as np


class Learner:
    def __init__(self, n_arms):
        """

        :param n_arms: number of arms of the environment that can be pulled
        :var t: current round
        :var rewards_per_arm: list containing the rewards collected when pulled the corresponding arm
        :var collected_rewards: nparray containing the rewards collected for each time step
        """
        self.n_arms = n_arms
        # current round:
        self.t = 0
        # empty list of n_arms elems:
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        # rewards collected at each round:
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """
        Update attributes of the learner storing the history of reward observations

        :param pulled_arm: arm pulled during the last time step
        :param reward: reward obtained from the pulled arm during the last time step
        """
        # append collected reward to list of rewards associated to the pulled arm:
        self.rewards_per_arm[pulled_arm].append(reward)
        # append collected reward to all the reward collected up to now:
        self.collected_rewards = np.append(self.collected_rewards, reward)
