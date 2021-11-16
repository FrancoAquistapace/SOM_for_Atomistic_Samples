# Self Organising Map for Clustering of Atomistic Samples
## Description
Self Organising Map (also known as Kohonen Network) implemented in Python for clustering of atomistic samples through unsupervised learning. The program allows the user to select wich per-atom quantities to use for training and application of the network, this quantities must be specified in the LAMMPS input file that is being analysed. The algorithm also requires the user to introduce some of the networks parameters:
- _f_: Fraction of the input data to be used when training the network, must be between 0 and 1.
- SIGMA: Maximum value of the _sigma_ function, present in the neighbourhood function.
- ETA: Maximum value of the _eta_ funtion, which acts as the learning rate of the network.
- N: Number of output neurons of the SOM, this is the number of groups the algorithm will use when classifying the atoms in the sample.
- Whether to use batched or serial learning for the training process.
- B: Batch size, in case the training is performed using batched learning.

The input file must be inside the same folder as the main.py file. Furthermore, the input file passed to the algorithm must have the LAMMPS dump format, or at least have a line with the following format:

`ITEM: ATOMS id x y z feature_1 feature_2 ...`

To run the software, simply execute the following command in a terminal (from the folder that contains the files and with a python environment activated):

`python3 main.py`

Check the software report in the general repository for more information: https://github.com/rambo1309/SOM_for_Atomistic_Samples_GeneralRepo

## Dependencies:
This software is written in Python 3.8.8 and uses the following external libraries:
- NumPy 1.20.1
- Pandas 1.2.4

(Both packages come with the basic installation of Anaconda)

## Updates:
V2 allows the user to analyse several files sequentally, using a single (previously specified) file for the training process. In this way, it allows to test whether transfer learning is possible or not for different samples. It is maintained in a different repository: https://github.com/rambo1309/SOM_for_Atomistic_Samples_V2

Currently working on giving the user the option to change the learning rate funtion, _eta_, with a few alternatives such as a power-law and an exponential decrease.
