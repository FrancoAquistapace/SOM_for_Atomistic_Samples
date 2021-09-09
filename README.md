# Self Organising Map for Clustering of Atomistic Samples
## Description
Self Organising Map (also known as Kohonen Network) implemented in Python for clustering of atomistic samples through unsupervised learning. The program allows the user to select wich per-atom quantities to use to train and apply the network, this quantities must be specified in the LAMMPS input file that is being analysed. The algorithm also requires the user to introduce some of the networks parameters:
- _f_: Fraction of the input data to be used when training the network, must be between 0 and 1.
- SIGMA: Maximum value of the _sigma_ function, present in the neighbourhood function.
- ETA: Maximum value of the _eta_ funtion, which acts as the learning rate of the network.
- N: Number of output neurons of the SOM, this is the number of groups the algorithm will use when classifying the atoms in the sample.
## Dependencies:
This software is written in Python 3.8.8 and uses the following external libraries:
- NumPy 1.20.1
- Pandas 1.2.4

(Both packages come with the basic installation of Anaconda)
