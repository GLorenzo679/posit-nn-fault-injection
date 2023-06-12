import os
import csv
import numpy as np

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def main():
    for i in range(50):
        fault_id = i

        tensor_index = (
            np.random.randint(0, 5),
            np.random.randint(0, 5),
            np.random.randint(0, 3), #64
            np.random.randint(0, 64),
        )
        
        with open(PATH + "/fault_list_layer_0.csv", "a+") as file:
            headers = [
                "fault_id",
                "tensor_index",
            ]

            writer = csv.DictWriter(file, delimiter=",", lineterminator="\n", fieldnames=headers)
            #writer.writeheader()
            writer.writerow(
                {
                    "fault_id": fault_id,
                    "tensor_index": tensor_index,
                }
            )

if __name__ == "__main__":
    main()