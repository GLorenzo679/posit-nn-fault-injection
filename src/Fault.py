import softposit as sp
import sys

class Fault:
    def __init__(self, fault_id, layer_index, tensor_index, bit_index, bit_value):
        self.fault_id = fault_id
        self.layer_index = layer_index
        self.tensor_index = tensor_index
        self.bit_index = bit_index
        self.bit_value = bit_value
        self.mask = sp.posit32()
        self.weight_start = sp.posit32()
        self.weight_corrupted = sp.posit32()

    def setWeightStart(self, weight_start, mask):
        # extract binary representation of np posit32
        np32_bin_representation = bin(int.from_bytes(weight_start.tobytes(), byteorder=sys.byteorder))
        # create a softposit posit32 with bits from np posit32
        self.weight_start.fromBits(int(np32_bin_representation, 2))
        self.mask.fromBits(int(mask, 0))

        self.weight_corrupted = self.weight_start ^ self.mask
        
    def printFault(self):
        print(str(self.fault_id) + ", " + str(self.layer_index) + ", " +
                str(self.tensor_index) + ", " + str(self.bit_index) + ", " + str(self.bit_value))
        

