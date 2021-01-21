import torch
import torch.nn as nn

# GMM implementation reference: https://stats.stackexchange.com/questions/120028/2-gaussian-mixture-model-inference-with-mcmc-and-pymc
# Neural Gas implementation reference: https://github.com/xyutao/fscil

class motion_probability(nn.Module):
    def __init__(self, state_shape, state_channel, motion_shape, device):
        super(motion_probability, self).__init__()
        self.state_shape = state_shape
        self.state_channel = state_channel
        self.motion_shape = motion_shape # 10
        self.device = device
        self.cnn_mid_channel = 16
        self.dropout= nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(dim=1)

        self.F1_DIM = 128
        self.F2_DIM = 64
        self.ReLU = nn.ReLU().to(self.device)
        self.Sigmoid = nn.Sigmoid().to(self.device)
        self.Tanh = nn.Tanh().to(self.device)
        self.CNN1 = nn.Conv2d(self.state_channel, self.cnn_mid_channel, 1, stride=3).to(self.device)
        self.CNN2 = nn.Conv2d(self.cnn_mid_channel, 3, 1, stride=3).to(self.device)
        self.MaxPool1 = nn.MaxPool2d(3, 1).to(self.device)

        self.F1 = nn.Linear(324, self.F1_DIM).to(self.device)
        self.F2 = nn.Linear(self.F1_DIM, self.F2_DIM).to(self.device)
        self.F3 = nn.Linear(self.F2_DIM, self.motion_shape).to(self.device)

    def forward(self, state):

        motion = self.CNN1(state)
        motion = self.dropout(motion)
        motion = self.CNN2(motion)
        motion = self.dropout(motion)
        motion = self.MaxPool1(motion)

        self.F1_DIM = motion.shape[0] * motion.shape[1] * motion.shape[2] * motion.shape[3]
        motion = motion.view(-1, self.F1_DIM)  # reshape Variable

        motion = self.F1(motion)
        motion = self.dropout(motion)
        motion = self.ReLU(motion)

        motion = self.F2(motion)
        motion = self.dropout(motion)
        motion = self.ReLU(motion)

        motion = self.F3(motion)
        motion = self.softmax(motion)

        return motion


# TODO : IMPLEMENT IT!
class state_probability():
    def __init(self):
        self.probability = None
        self.state_prob_func = None
        self.pdf = None

    def set_probability(self, dataset):
        # TODO: implement pdf with self.state_prob_func
        self.pdf = dataset

    def get_probability(self, state):
        self.probability = self.state_prob_func(state)
        return self.probability
