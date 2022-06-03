
Note: The code is in both .ipynb and HTML format. HTML format can be opened to read the code.


The main files that implement the theme of the paper is "Target Network.ipynb" which implements the source to target transfer of parameters along with knowlege transfer part of the paper.
"Source Network Training" Trains the source network
"Data Processing Source and Target" basically processes the dataset to balance it and get the target and source examples for training
"Data Prcessing" Takes raw data from the RealDisp dataset files and makes a dictionary for each activitiy and each subject 
"data_handling" Mainly takes the dictionary made earlier and preprocesses the data to convert data into 5 second windows along with further preping the data for further data distribution for source and target data.
"utils.py" is a utility code file