# DE

import numpy as np
import torch
import probability

class DataFilter():
    def __init__(self, pm, pb, po):
        self.pm = pm
        self.pb = pb
        self.po = po

        self.pm_threshold = 0.5
        self.pb_threshold = 0.5
        self.po_threshold = 0.5

        self.pmbo_threshold = 0.7

    def is_novel(self, online_state, online_behavior, online_motion):

        pm_prob = self.pm(online_motion, online_behavior, online_state)
        pb_prob = self.pm(online_behavior, online_state)
        po_prob = self.pm(online_state)

        if  pm_prob * pb_prob * po_prob > self.pmbo_threshold:
            return True
        else:
            return False