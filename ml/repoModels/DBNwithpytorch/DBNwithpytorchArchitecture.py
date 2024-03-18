#STILL CANT USE THIS MODEL, TQDM
#Use model with access to DBN.py file and RBM.py
#This is an example of how to access and train the DBN with pytorch: 
layers = [7, 5, 2]
dbn = DBN(10, layers)
dbn.train_DBN(dataset)