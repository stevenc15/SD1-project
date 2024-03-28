# Use model with access to models(numpy).py for numpy implementation, 
# and utils.py, activations.py
# if you want tensorflow implementation, access to models(tf).py, for this implementation, 
# you need access to both models(tf).py and models(numpy).py, along with utils.py, activations.py
# This is an example of how to access and train the DBN: 

#Method to initialize model
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)

#Method to train model
classifier.fit(X_train, Y_train)