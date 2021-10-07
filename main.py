#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:01:47 2021

@author: francoaquistapace

Copyright 2021 Franco Aquistapace

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

# Import modules and SOM
import pandas as pd
import time

from SOM import *

# Fixed seed for consistent results
np.random.seed(1982) 

# Open file and extract column names
path = input('Insert file path: ', )
print('Reading file...')
#path = 'dump.ensayo.2900000.config'
file = open(path,'r')
found_cols = False
header_lines = []
line_count = 0
while not found_cols:
    line = file.readline()
    line_count += 1
    if 'ITEM: ATOMS' in line:
        columns = line.split()
        columns.pop(0) # Pop 'ITEM:'
        columns.pop(0) # Pop 'ATOMS'
        found_cols = True
    header_lines.append(line)
file.close()

# Build DataFrame from data file
print('Importing data...')
df = pd.read_csv(path,sep=' ', 
                 skiprows=line_count, names=columns)

# Show columns so that the user can select the features
print('\nAvailable columns: ')
print(columns)


# Select features, could improve on this in the future
features = []
message = 'Insert a feature name to include, \'exit\' to quit the program' +\
         ' or press enter to continue: '
print('\n(If no feature is selected, '+\
      'all available columns are taken as features)\n')
key = input(message, )
while key != '':
    # Check if user wants to exit
    if key == 'exit':
        print('Process terminated')
        exit()
        
    # Check that the input matches one of the column names and that
    # it hasn't already been added to the features
    if key in columns and not key in features:
        features.append(key)
    else:
        print('Error: Feature is already selected or not available')
    
    key = input(message, )

# If the user pressed enter directly we use all the columns as features
if len(features) == 0:
    features = columns
    print('\nUsing all columns as features\n')
else:
    print('\nUsing features: %s\n' % features)

# Normalize features
norm_df = df[features].copy()
for feat in features:
    min_value = norm_df[feat].min()
    max_value = norm_df[feat].max()
    norm_df[feat] = (norm_df[feat] - min_value)/ (max_value - min_value)
    
# Shuffle for training
frac = input('Insert fraction of the data to be used for '+\
             'training, must be between 0 and 1: ',)
if frac == '1':
    frac = int(1)
else:
    frac = float(frac)
training_df = norm_df.sample(frac=frac)


# Initialize SOM parameters
print('\nPredetermined SOM parameters are:' + '\nSIGMA = 1' + '\nETA = 0.5')
SIGMA = input('Insert new SIGMA parameter or press enter to pass: ',)
if SIGMA == '':
    SIGMA = 1
else:
    SIGMA = float(SIGMA)
ETA = input('Insert new ETA parameter or press enter to pass: ',)
if ETA == '':
    ETA = 0.5
else:
    ETA = float(ETA)
    
GROUPS = int(input('Insert number of clusters: ',))
SIZE = (len(features),GROUPS)
# Build SOM model
som = SOM(sigma=SIGMA, eta=ETA, size=SIZE)

# Start timing...
time1 = time.time()


# Train SOM and predict groups
print('\nTraining SOM...')
som.train(training_df)
print('Predicting atom groups...')
results = som.predict(norm_df)
# We only need the last column, which contains the grouping result
result_cols = results.columns.to_list()
groups = results[result_cols[-1]]

# Concat new DataFrame
new_df = pd.concat([df,groups], axis=1)


# Save new file with the group assigned to each atom
print('Writing results...')
new_path = 'SOM_' + path
new_file = open(new_path, 'w')

# Write the header of the file
for i in range(len(header_lines)):
    line = header_lines[i]
    if i == len(header_lines) - 1:
        line = line.replace('\n', ' som_cluster\n')
    new_file.write(line)

# Now let's write the new data
final_string = new_df.to_csv(index=False, sep=' ', 
                         float_format='%s', header=False)
new_file.write(final_string)

    
new_file.close()

# Finish timing
time2 = time.time()

print('Process completed')
print('Elapsed time: ' + str(round(time2-time1,3)) + ' seconds')
    

