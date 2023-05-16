import numpy as np
from scipy.stats import norm
from src.Fault import Fault


class Injection:
    def __init__(self):
        self.fault_list = []
        self.type = None
        self.num_bit_representation = None

    def __compute_binary_mask__(self, fault):
        return "0b" + "0" * (self.num_bit_representation - 1 - fault.bit_index) + str(1) + "0" * (fault.bit_index)

    def create_injection_list(
        self, num_weight_net, num_layer, tensor_shape, num_bit_representation, type, number_of_faults
    ):
        self.num_bit_representation = num_bit_representation
        self.type = type

        t = compute_t(0.8)
        N = num_weight_net * num_bit_representation * 2

        if number_of_faults == 0:
            number_of_faults = int(compute_date_n(N, 0.5, 0.01, t))

        print(f"Number of faults to apply: {number_of_faults}")

        for i in range(number_of_faults):
            fault_id = i
            layer_index = np.random.randint(0, num_layer)
            tensor_index = (
                np.random.randint(0, tensor_shape[0]),
                np.random.randint(0, tensor_shape[1]),
                np.random.randint(0, tensor_shape[2]),
                np.random.randint(0, tensor_shape[3]),
            )
            bit_index = np.random.randint(0, num_bit_representation)

            fault = Fault(fault_id, layer_index, tensor_index, bit_index, self.type)
            fault.binary_mask = self.__compute_binary_mask__(fault)
            self.fault_list.append(fault)

    def print_injection_list(self):
        for i in range(len(self.fault_list)):
            self.fault_list[i].print_fault()


def compute_t(confidence_level: float = 0.8):
    """
    Compute the t value from the confidence level
    :param confidence level: The desired confidence level
    :return: The t corresponding to the input confidence level
    """
    alpha = (1 - confidence_level) / 2
    t = -norm.ppf(alpha)
    return t


def compute_date_n(N: int, p: float = 0.5, e: float = 0.01, t: float = 2.58):
    """
    Compute the number of faults to inject according to the DATE23 formula
    :param N: The total number of parameters. If None, compute the infinite population version
    :param p: Default 0.5. The probability of a fault
    :param e: Default 0.01. The desired error rate
    :param t: Default 2.58. The desired confidence level
    :return: the number of fault to inject
    """
    if N is None:
        return p * (1 - p) * t**2 / e**2
    else:
        return N / (1 + e**2 * (N - 1) / (t**2 * p * (1 - p)))
