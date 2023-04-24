'''N e': numero totale di pesi nella rete * numero di bit nella rappresentazione * 2 (possibili valori assunti da un singolo bit).
Ricapitolando, la vostra fault list deve contenere n faults (calcolati con __compute_date_n e __compute_t, usando una confidence_level=0.8 e un e=0.01). 
Ogni fault e' rappresentato da:
fault_id, layer_index, tensor_index, bit_index, bit_value
Dove layer_index e' l'indice del layer colpito da un fault: per ora iniettiamo solo nei pesi dei layer convoluzionali della rete. Tensor_index e' l' indice all'interno di un tensore dei pesi, bit_index e' l'indice di un bit all'interno della rappresentazione binaria di un peso e bit_value e' il valore a cui settare quel bit.
'''
import  Injection

def maskGenerator(numBitRapresentation, bit_index, bit_value):
    mask = '0b' + '0'*bit_index + str(bit_value) + '0'*(numBitRapresentation - bit_index - 1)
    print(f"numBit = {numBitRapresentation}, bitIndex = {bit_index}, bit value = {bit_value}, mask = {mask}")

if __name__ == "__main__":
    numWeightNet = 10
    numBitRapresentation = 8
    valueBit = 2
    numLayer = 5
    numBatch = 3
    batchHeight = 2 
    batchWidth = 4
    batchFeatures = 5 
    #injection = Injection.Injection()
    #injection.createInjectionList(numWeightNet, numBitRapresentation, valueBit, numLayer, numBatch, batchHeight, batchWidth, batchFeatures)
    #injection.printInjectionList()
    maskGenerator(8, 5, 1)
    maskGenerator(8, 2, 1)