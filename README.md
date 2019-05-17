# Deep Learning Projects
Deep Learning (EE-559 @EPFL) Spring 2019

# [Project 1 - Weight Sharing and Auxiliary Loss](Proj1)
```
cd Proj1
```
## How to run the code
### Dependencies
Following libraries are required:
```
matplotlib
pandas
pickle
torch
torchvision
seaborn
```
They can be installed using the following command:
```
pip install <insert lib here>
```
### Test
In order to test the results claimed in the report, please run the following `test.py` file using the following command:
```
python3 test.py
```
One can use its own parameters either by modifying the `constants.py` file or by using the following command line arguments:
```
usage: test.py [-h] [--models MODELS] [--epochs EPOCHS] [--rounds ROUNDS]
               [--seed SEED] [--aux AUX_LOSS_FACTOR]
```

# [Project 2 - Deepy](Proj2)
```
cd Proj2
```
## Dependencies
Following libraries are required:
```
torch
```
They can be installed using the following command:
```
pip install <insert lib here>
```
## Example
An simple example is provided in the report and a full example is provided in the file `test.py`
 which can be used with the following command:.
```
python3 test.py
```
