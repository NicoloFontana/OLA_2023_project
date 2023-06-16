from learner import *
import parameters as param


class TSLearner(Learner):
    def __init__(self, n_arms):
        """
        Implements a learner following the Thompson Sampling algorithm

        :var beta_parameters: parameters for the Beta distribution used to draw the random sample
        """
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self):
        """
        Choose the arm to pull as the one giving the highest random sample from the corresponding Beta distribution

        :return: number of the arm to pull
        """
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])*(param.prices-param.cost))
        return idx

    def update(self, pulled_arm, reward):
        """
        Update history of observations and parameters of the Beta distribution corresponding to the pulled arm

        :param pulled_arm: number of the arm pulled at the last time step
        :param reward: (binary) reward obtained at the last time step
        """
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        # Add the success (if any) to alpha
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward[0]
        # Add the failure (if any) to beta (1-rew=1 iff rew=0 iff fail)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + reward[1]

    def update_bulk(self, pulled_arms, rewards):
        for i, arm in enumerate(pulled_arms):
            self.update(arm, rewards[i])


