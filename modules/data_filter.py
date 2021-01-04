# DE

import numpy as np
# import torch
# import probability

class DataFilter():
    def __init__(self):
        self.pm = None
        self.pb = None
        self.po = None

        self.pm_threshold = 0.5
        self.pb_threshold = 0.5
        self.po_threshold = 0.5

        self.pmbo_threshold = 0.7

    def is_novel(self, online_state, online_motion):

        # pm_prob = self.pm(online_data[0], online_data[1])
        # po_prob = self.pm(online_data[0])

        random = np.random.uniform(0, 1)

        # if  pm_prob * pb_prob * po_prob > self.pmbo_threshold:
        if random > 0.5:
            return True
        else:
            return False