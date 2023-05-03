from scipy.stats import norm
import random
from Fault import Fault


class Injection:
    def __init__(self):
        self.fault_list = []

    def createInjectionList(self, num_weight_net, num_bit_representation, num_layer, num_batch, batch_height, batch_width, batch_features):
        self.num_bit_representation = num_bit_representation
        t = compute_t(0.8)
        N  = num_weight_net * num_bit_representation * 2
        number_of_faults = int(compute_date_n(N, 0.5, 0.01, t))
        print(f"Number of faults to apply: {number_of_faults}")

        for i in range(number_of_faults):
            fault_id = i
            layer_index = random.randrange(0, num_layer)
            tensor_index = (random.randrange(0, num_batch),
                            random.randrange(0, batch_height),
                            random.randrange(0, batch_width),
                            random.randrange(0,batch_features))
            bit_index = random.randrange(0, num_bit_representation)
            bit_value = 1
            fault = Fault(fault_id, layer_index, tensor_index, bit_index, bit_value)
            self.fault_list.append(fault)


    def printInjectionList(self):
        for i in range(len(self.fault_list)):
            self.fault_list[i].printFault()    


def compute_t(confidence_level: float = 0.8):
    """
    Compute the t value from the confidence level
    :param confidence level: The desired confidence level
    :return: The t corresponding to the input confidence level
    """   
    alpha = (1 - confidence_level) / 2
    t = -norm.ppf(alpha)
    return t


def compute_date_n(N: int,
                     p: float = 0.5,
                     e: float = 0.01,
                     t: float = 2.58):
    """
    Compute the number of faults to inject according to the DATE23 formula
    :param N: The total number of parameters. If None, compute the infinite population version
    :param p: Default 0.5. The probability of a fault
    :param e: Default 0.01. The desired error rate
    :param t: Default 2.58. The desired confidence level
    :return: the number of fault to inject
    """
    if N is None:
        return p * (1-p) * t ** 2 / e ** 2
    else:
        return N / (1 + e ** 2 * (N - 1) / (t ** 2 * p * (1 - p)))





