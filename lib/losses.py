import keras.backend as K



# ================ LABELED LOSSES ================
def binary_crossentropy(y_true, y_pred):
    return K.binary_crossentropy(y_pred, y_true)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true)

def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def modified_hinge(y_true, y_pred):
    return K.mean(K.maximum(.51 - y_true * y_pred, 0.), axis=-1)
	
def modified_hinge2(y_true, y_pred):
    return K.mean(K.square(K.maximum(.51 - y_true * y_pred, 0.)), axis=-1)
	

def mixed_loss(y_true, y_pred):
	return K.categorical_crossentropy(y_pred, y_true) + K.mean(K.maximum(.5 - y_true * y_pred, 0.), axis=-1)
	
def minimize_output2(y_true, y_pred):
	return  K.square(y_pred)

def maximize_output(y_true, y_pred):
	return - y_pred

# if the loss is modeled by the network
def minimize_output(y_true, y_pred):
	return y_pred

# ================ UNLABELED LOSSES (y_true=[1,...,1])================

# General Label Entropy (increase scoring variance)
def label_entropy(y_true, y_pred):
	return  - K.sum(y_pred * K.log(y_pred), axis=-1)













