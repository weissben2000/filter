# Filtering algorithm

Pipeline to run all codes: pre_processing -> neuralNetwork -> data_reduction -> plotting & turnOnCurves.

pre_processing.ipynb reads all datasets from a given location (say cmslpc or lxplus), and shuffles and separates them into training and testing datasets, and saves the corresponding output files locally. These files will then be sent as input to neuralNetwork.ipynb. Note - A set of files is produced for one pT boundary; this file has to be run once for every pT boundary. This was set up in a rather cumbersome way owing to requiring checks on the truth distributions and event numbers at every pT boundary. 
NOTE: pre-processing.py can now run for all pT boundaries at one go and save validation/check information for every pT boundary!

neuralNetwork.ipynb trains and evaluates the model on the test and train datasets, respectively.

data_reduction.ipynb quantifies the model's performance (i.e., calculate signal efficiency, background rejection, and data reduction).

The remaining files help produce the relevant performance results and plots.
