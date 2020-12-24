# train and update LPNet

from modules.bmnet import BMNet
from modules.data_filter import DataFilter
from modules.databatch_composer import DataBatchComposer
from modules.dataset_supervisor import DatasetSupervisor
from modules.data_exchanger import DataExchanger
import carla

from carla.run import CarlaWorld
from datetime import datetime

def get_current_time():
    now = datetime.now()
    # now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
    # now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

    # return now, now_date, now_time
    return now

def train_bmnet(bmnet, data):
    output = bmnet(data)

    # TODO: train bmnet with for loop and loss + backward

    return bmnet

def main():
    start_time = get_current_time()
    current_time = get_current_time()
    # TODO: Fill in pm, pb, po
    data_filter = DataFilter()
    data_exchanger = DataExchanger()
    bmnet = BMNet()
    forever = True
    # TODO: change it to carlaworld
    env = None

    print("INCREMENTAL INTELLIGENCE SYSTEM OPERATING...")
    while forever:
        # Collect novel online data in daytime.
        print("START COLLECTING ONLINE DATA...")
        current_state = env.reset()
        while current_time - start_time < 2 is True:

            predicted_behavior, predicted_motion = bmnet(current_state)
            next_state, info = env.step(predicted_motion)

            if info["disengagement"] is True:
                # record it then could make it runable afterward by expert human
                return

            if data_filter.is_novel(current_state, predicted_behavior, predicted_motion):
                data_exchanger.exchange([current_state, predicted_behavior, predicted_motion])

            current_state = next_state
            current_time = get_current_time()

        print("END COLLECTING ONLINE DATA...")

        # TODO: Make dataset
        dataset = ""
        updated_bmnet = train_bmnet(bmnet, dataset)

        # TODO: Save updated bmnet model
        updated_bmnet_dir = "./trained_model/"

        # TODO: Update bmnet
        bmnet = updated_bmnet

    return 0

if __name__ == '__main__':
    main()