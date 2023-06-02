from learner import *

class GreedyLearner(Learner):
    def __init__(self, n_arms):
        """
        Implements a greedy learner (ie always pull the arm with best expected reward)

        :var expected_rewards: nparray containing the expected reward for each arm
        """
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        """
        Choose the arm to pull as the one with the highest expected reward

        :return: number of the arm pulled
        """
        # Pull each arm at least one time
        if(self.t < self.n_arms):
            return self.t
        # Pull arm with max expected reward
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        """
        Update the history of observation and the expected rewards

        :param pulled_arm: number of the arm pulled
        :param reward: (binary) reward obtained at the last time step
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm]*(self.t-1)+reward)/self.t