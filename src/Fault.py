class Fault:
    def __init__(self, fault_id, layer_index, tensor_index, bit_index, bit_value):
        self.fault_id = fault_id
        self.layer_index = layer_index
        self.tensor_index = tensor_index
        self.bit_index = bit_index
        self.bit_value = bit_value
        self.weight_start = None
        self.weight_corrupted = None

    def setWeightStart(self, weight_start, mask):
        self.weight_start = weight_start
        self.weight_corrupted = self.weight_start ^ mask
        
    def printFault(self):
        print(str(self.fault_id) + ", " + str(self.layer_index) + ", " +
                str(self.tensor_index) + ", " + str(self.bit_index) + ", " + str(self.bit_value))
        

