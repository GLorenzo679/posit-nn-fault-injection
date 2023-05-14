# posit-nn-fault-injection
Posit NN fault injection. Code developed during the "Architetture dei sistemi di elaborazione" special project.

# Project structure

This project is structured as follows:

## Project tree
```
SP5/
├─ data/
│  ├─ dataset_name_1/
│  │  ├─ model_name_1/
│  │  ├─ model_name_2/
│  ├─ dataset_name_2/
├─ models/
├─ res/
│  ├─ dataset_name_1/
│  │  ├─ model_name_1/
│  │  ├─ model_name_2/
│  ├─ dataset_name_2/
├─ src/
├─ utils/
├─ main.py
```

## Folder content
```
data/ -> Containes the weights obtained in the training phase. Each dataset and model has its own folder
models/ -> Contains the custom models used
res/ -> Containes the results of the FI campaign. Each dataset and model has its own folder
src/ -> Contains the files that manages the FI campaigns
utils/ -> Folder with utility files (parser, parameters getters, ...)
main.py -> Main file to perform FI campaigns and inference
```

# Setup
## Prerequisite:
- Python 3.6 (strictly required)
- protobuf 3.19.6 (strictly required)
```
pip install protobuf==3.19.6
```
In order to use this FI framework the following python packages are required:

- numpy 1.15.2
- softposit
- numpy-posit (a modified version of numpy supporting posit data type)
- tensorflow-posit (a modified version of tensorflow supporting posit data type)
- scipy

You can install these packages with `pip`, using the following commands (the creation of a virtual env is recommended):

```
pip install requests numpy==1.15.2 softposit

pip install numpy-posit

pip install https://s3-ap-southeast-1.amazonaws.com/posit-speedgo/tensorflow_posit-1.11.0.0.0.1.dev1-cp36-cp36m-linux_x86_64.whl

pip install scipy
```

The order of the commands is important.

For more detailed instructions you can check [Deep PeNSieve installation guide](https://github.com/RaulMurillo/deep-pensieve/blob/master/README.md#installation).

# Usage
To run the programm it is necessary set some parameters from command line. 
Required values are:
```
--type or -t                #Numeric format:  posit8/posit16/posit32/float32        
--network-name or -n        #Network to use:  convnet
--data-set or -d            #Input dataset:   CIFAR10     
--bit-len or -b             #Number of bit:   8/16/32/32
                            #It is critical that this value be consistent with the numeric format
```

Other commands may be optional because they use some default values, but they can be modified using: 
```
--batch-size                #Batch size dimension  
--size or -s                #Test set size
--force-n                   #Force to a specific number of injections
--seed                      #Seed to random value
```

To watch more details and its default value you can run the command:
```
--help or -h
```

# Output

# Credits
The ML tasks performed in this framework rely on: [Deep PeNSieve](https://github.com/RaulMurillo/deep-pensieve/).