from __future__ import print_function
import keras
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from convca_read import prepare_data_as, prepare_template, normalize
import math
from math import pi
import numpy as np

from keras import backend as K
from keras.layers import Layer

## correlation analysis layer
class Corr(Layer):

    def __init__(self, params, **kwargs):
        self.tw = params['tw']
        self.Fs = params['Fs']
        self.cl = params['cl']
        self.corr = None
        super(Corr, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable weight variable for this layer.
        super(Corr, self).build(input_shape)

    def call(self, input, **kwargs):
        x = input[0] # [?,tw] signal
        t = input[1] # [?,tw,cl] reference
        t_ = K.reshape(t,(-1,self.tw,self.cl))

        corr_xt = K.batch_dot(x,t_,axes=(1,1)) # [?,cl]
        corr_xx = K.batch_dot(x,x,axes=(1,1)) # [?,1]
        corr_tt = K.sum(t_*t_,axis=1) # [?,cl]
        self.corr = corr_xt/K.sqrt(corr_tt)/K.sqrt(corr_xx)
        self.out = self.corr
        return self.out

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.out)


## parameters
# channels: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2
permutation = [47,53,54,55,56,57,60,61,62]
params = {'tw':50,'Fs':250,'cl':40,'ch':len(permutation)}

for subj in range(35,36):
    print(subj)
    train_run = [1,2,3,4,5]
    test_run = [0]

    ## build dataset
    x_train,y_train,freq = prepare_data_as(subj,train_run,params['tw']) ## [?,tw,ch]
    x_test,y_test,__ = prepare_data_as(subj,test_run,params['tw']) # [?,tw,ch]
    x_train = x_train.reshape((x_train.shape[0],params['tw'],params['ch'],1))
    x_test = x_test.reshape((x_test.shape[0],params['tw'],params['ch'],1))
    y_train = keras.utils.to_categorical(y_train, params['cl'])
    y_test = keras.utils.to_categorical(y_test, params['cl'])

    ## build reference signal
    template = prepare_template(subj,train_run,params['tw']) # [cl*sample,cl,tw,ch]
    template = np.transpose(template, axes=(0,2,1,3)) # [cl*sample,tw,cl,ch]

    ## build model
    # signal-CNN
    signal = Input(shape=(params['tw'],params['ch'],1))
    conv11 = Conv2D(16,(9,9),padding='same')(signal)
    conv12 = Conv2D(1,(1,9),padding='same')(conv11)
    conv13 = Conv2D(1,(1,9),padding='valid')(conv12)
    drop1 = Dropout(0.75)(conv13)
    flat1 = Flatten()(drop1)

    # reference-CNN
    temp = Input(shape=(params['tw'],params['cl'],params['ch']))
    conv21 = Conv2D(40,(9,1),padding='same')(temp)
    conv22 = Conv2D(1,(9,1),padding='same')(conv21)
    drop2 = Dropout(0.15)(conv22)

    # correlation layer
    corr = Corr(params)([flat1,drop2])

    # dense layer for classification
    out = Dense(params['cl'],activation='softmax')(corr)
    model = Model(inputs=[signal,temp],outputs=out)


    opt = keras.optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, clipvalue=5.)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    print(model.summary())

    # train & test
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    model.fit([x_train,np.tile(template, (len(train_run),1,1,1))], y_train, batch_size=32, epochs=100,
        validation_data=([x_test,template], y_test), shuffle=True)

    # Score trained model.
    scores = model.evaluate([x_test,template], y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
