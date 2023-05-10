import sys


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
        # create a softposit posit with bits from np posit
        self.weight_start.fromBits(int(np_bin_representation, 2))

        # print(f"np bin repr: {np_bin_representation}")
        # print(f"weight start: {self.weight_start}")

        self.mask.fromBits(int(self.binary_mask, 0))

        print(f"binary_mask: {self.binary_mask}")

        self.weight_corrupted = self.weight_start ^ self.mask

    #def print_fault(self):
    #    print(
    #        str(self.fault_id)
    #        + ", "
    #        + str(self.layer_index)
    #        + ", "
    #        + str(self.tensor_index)
    #        + ", "
    #        + str(self.bit_index)
    #        + ", "
    #        + str(self.bit_value)
    #    )
