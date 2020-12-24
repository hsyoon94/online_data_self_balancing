# train and update LPNet

from modules.bmnet import BMNet
from modules.data_filter import DataFilter
from modules.databatch_composer import DataBatchComposer
from modules.dataset_supervisor import DatasetSupervisor
from modules.data_exchanger import DataExchanger
import carla

import carla.run as run
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

    print("INCREMENTAL INTELLIGENCE SYSTEM OPERATING...")
    while forever:
        # Collect novel online data in daytime.
        print("START COLLECTING ONLINE DATA...")
        while current_time - start_time < 2 is True:

            online_data = run.get_state(bmnet)
            predicted_behavior, predicted_motion = bmnet(online_data)
            if data_filter.is_novel(online_data, predicted_behavior, predicted_motion):
                data_exchanger.exchange([online_data, predicted_behavior, predicted_motion])

            current_time = get_current_time()

        print("END COLLECTING ONLINE DATA...")

        # TODO: Make dataset
        dataset = ""
        updated_bmnet = train_bmnet(bmnet, dataset)

        # TODO: Save updated bmnet model
        updated_bmnet_dir = ""

        # TODO: Update bmnet
        bmnet = updated_bmnet

    return 0

if __name__ == '__main__':
    main()