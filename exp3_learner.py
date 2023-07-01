import parameters
from learner import *
import random
from math import sqrt,log

class Exp3Learner(Learner):
    def __init__(self, n_arms, upperbound_total_reward=1.0, rewardMin=0.0, rewardMax=1.0):
        """
        Implements a learner following the EXP3 algorithm

        :var gamma: parameters for the ...
        """
        super().__init__(n_arms)
        self.weights = np.ones(n_arms)
        self.rewardMin = rewardMin  # the min of all the price phases
        self.rewardMax = rewardMax  # the max of all price phases
        upperbound_total_reward_scaled = (upperbound_total_reward - self.rewardMin) / (self.rewardMax - self.rewardMin)
        exponential = 2.7182818284
        self.gamma = min(1.0, sqrt(n_arms*log(n_arms)/(upperbound_total_reward_scaled*(exponential-1))))
        #self.gamma = np.sqrt(2*log(n_arms)/(n_arms*upperbound_total_reward))
        #self.gamma=0.1
        print(f'Exp3 gamma: {self.gamma} with ')


    def pull_arm(self):
        """
        Choose the arm to pull as a random draw form  a discrete distribution

        :return: number of the arm to pull
        """

        probabilityDistribution = distr(self.weights, self.gamma)
        idx = draw(probabilityDistribution)
        return idx

    def update(self, pulled_arm, reward):
        """
        Update history of observations and weights of the arm

        :param pulled_arm: number of the arm pulled at the last time step
        :param reward: (binary) reward obtained at the last time step
        """
        self.t += 1

        #scaledReward = (reward[0]/(reward[0]+reward[1]) - self.rewardMin) / (self.rewardMax - self.rewardMin)  # rewards scaled to 0,1
        scaledReward = ( (reward[0]/(reward[0]+reward[1])*(parameters.prices[pulled_arm] - parameters.cost) )- self.rewardMin) / (self.rewardMax - self.rewardMin)  # rewards scaled to 0,1

        probabilityDistribution = distr(self.weights, self.gamma)
        estimatedReward = 1.0 * scaledReward / probabilityDistribution[pulled_arm]
        self.weights[pulled_arm] *= np.exp(estimatedReward * self.gamma / self.n_arms)  # important that we use estimated reward here!
        self.update_observations(pulled_arm, reward[2])


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).


def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1

def distr(weights, gamma=1.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)

