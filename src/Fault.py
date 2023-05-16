import struct
import sys

import numpy as np


class Fault:
    def __init__(self, fault_id, layer_index, tensor_index, bit_index, type):
        self.fault_id = fault_id
        self.layer_index = layer_index
        self.tensor_index = tensor_index
        self.bit_index = bit_index
        self.binary_mask = None
        self.mask = type()
        self.weight_start = type()
        self.weight_corrupted = type()

    def set_weight(self, np_weight_start):
        # extract binary representation of np posit
        np_bin_representation = bin(int.from_bytes(np_weight_start.tobytes(), byteorder=sys.byteorder))

        if type(np_weight_start) is not np.float32:
            # create a softposit posit with bits from np posit
            self.weight_start.fromBits(int(np_bin_representation, 2))

            # create sp mask from binary string
            self.mask.fromBits(int(self.binary_mask, 0))
            print(f"binary_mask: {self.binary_mask}")

            # bitwise xor
            self.weight_corrupted = self.weight_start ^ self.mask
        else:
            # save current float weight
            self.weight_start = np_weight_start

            # create float mask from binary string
            self.mask = np.float32(struct.unpack("f", struct.pack("I", int(self.binary_mask, 0)))[0])
            print(f"binary_mask: {self.binary_mask}")

            # --- ONLY FOR DEBUG ---

            # bitwise xor to get binary string
            # weight_corrupted_bit = bin(
            #     int.from_bytes(
            #         struct.pack(
            #             "I",
            #             int.from_bytes(np_weight_start.tobytes(), byteorder=sys.byteorder) ^ int(self.binary_mask, 0),
            #         ),
            #         byteorder=sys.byteorder,
            #     )
            # )
            # print(f"np_bin_reprensentation:{np_bin_representation}")
            # print(f"np_weight_corrupted:{weight_corrupted_bit}")

            # --------------------------------

            # perform binary xor and create corrupted float
            self.weight_corrupted = np.float32(
                struct.unpack(
                    "f",
                    struct.pack(
                        "I",
                        int.from_bytes(np_weight_start.tobytes(), byteorder=sys.byteorder) ^ int(self.binary_mask, 0),
                    ),
                )[0]
            )

    def print_fault(self):
        print(
            str(self.fault_id)
            + ", "
            + str(self.layer_index)
            + ", "
            + str(self.tensor_index)
            + ", "
            + str(self.bit_index)
            + ", "
            + str(self.bit_value)
        )
