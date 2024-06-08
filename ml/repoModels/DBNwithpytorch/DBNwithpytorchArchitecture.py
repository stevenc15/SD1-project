#STILL CANT USE THIS MODEL, TQDM not known
#Use model with access to DBN.py file and RBM.py
#This is an example of how to access and train the DBN with pytorch: 

#define layers
layers = [7, 5, 2]

#method to initialize model
dbn = DBN(10, layers)

#method to train model
dbn.train_DBN(dataset)