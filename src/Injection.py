from scipy.stats import norm
import random
from . import Fault


class Injection:
    def __init__(self):
        self.faultList = []

    def createInjectionList(self, numWeightNet, numBitRapresentation, valueBit, numLayer, numBatch, batchHeight, batchWidth, batchFeatures):
        self.numBitRapresentation = numBitRapresentation
        t = compute_t(0.8)
        N  = numWeightNet * numBitRapresentation * valueBit
        numberOfFaults = int(compute_date_n(N, 0.5, 0.01, t))
        print(f"Number of faults to apply:\n{numberOfFaults}")
        for i in range(numberOfFaults):
            fault_id = i
            layer_index = random.randrange(0, numLayer)
            tensor_index = (random.randrange(0, numBatch),
                            random.randrange(0, batchHeight),
                            random.randrange(0, batchWidth),
                            random.randrange(0,batchFeatures)
                            )
            bit_index = random.randrange(0, numBitRapresentation)
            bit_value = random.randrange(0, 2)
            fault = Fault(fault_id, layer_index, tensor_index, bit_index, bit_value)
            self.faultList.append(fault)


    def printInjectionList(self):
        for i in range(len(self.faultList)):
            self.faultList[i].printFault()    


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





