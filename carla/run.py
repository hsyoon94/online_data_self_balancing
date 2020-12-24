# run.

import carla

class CarlaWorld():
    def __init__(self):
        self.world = carla.world()
        self.current_state = None
        self.next_state = None
        self.info = None

    def reset(self):
        self.state = None

    def step(self, motion):
        self.motion = motion

        self.info["disengagement"] = True

        return self.next_state, self.info

