import numpy as np
import torch

# GMM implementation reference: https://stats.stackexchange.com/questions/120028/2-gaussian-mixture-model-inference-with-mcmc-and-pymc
# Neural Gas implementation reference: https://github.com/xyutao/fscil

class motion_probability():
    def __init(self):
        self.probability = None
        self.motion_gmm = None
        # TODO: GMM
        self.pdf = None

    def set_probability(self, dataset):
        # TODO: implement pdf with gmm
        self.pdf = dataset

    def get_probability(self, state, behavior, motion):
        self.probability = self.motion_gmm(motion, behavior, state)
        return self.probability


class behavior_probability():
    def __init(self):
        self.probability = None
        self.behavior_gmm = None
        # TODO: GMM
        self.pdf = None

    def set_probability(self, dataset):
        # TODO: implement pdf with gmm
        self.pdf = dataset

    def get_probability(self, state, behavior):
        self.probability = self.behavior_gmm(behavior, state)
        return self.probability


class state_probability():
    def __init(self):
        self.probability = None
        # TODO : Maybe Neural Gas?
        self.state_prob_func = None
        self.pdf = None

    def set_probability(self, dataset):
        # TODO: implement pdf with self.state_prob_func
        self.pdf = dataset

    def get_probability(self, state):
        self.probability = self.state_prob_func(state)
        return self.probability
