#Notebook in which model is made and used is
#found under the file "NeuroleapModelArchitecture.py"

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(8,input_dim = train_X.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256,activation='relu'))
NN_model.add(Dense(256,activation='relu'))

# The Output Layer :
NN_model.add(Dense(66,activation='linear'))

# Compile the network :
NN_model.compile(loss='mse', optimizer='Adam')