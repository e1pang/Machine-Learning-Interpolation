import numpy as np
from numpy import argmax
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
from generateData import bezier_lstm_separate, generate_control
'''
-LSTM cells, encoder/decoder and teacher forcing
-train TWO SEPERATE models: one for x-coord and another for y-coord: WORKED
then use them together to plot the path of the mouse from point 1 to 2
    -model input: start and end coords; output: coords in path 

-using categorical input/output did not work as well as cardinality increased
    -continuous data worked better

-observations:
    1) sometimes the training gets stuck in a local optima, have to restart
    2) interestingly, the ml created data sometimes follows the training data
    and other times mirrors the curve

-screenshots, it is most likely possible to do better...
-red is the model output, blue is the training data
-continuous path with 10 pts
https://gyazo.com/d17c2a51f131d964bcbd6d2b9b84e2b9
https://gyazo.com/3e250c672a45aa9156795da257b940fc
https://gyazo.com/c44564fa10d570e3898a320f2e7aae6d
https://gyazo.com/2eaaace48571062b52d606bed6629e9d
https://gyazo.com/2abef7df74fc70b9f5c45c2a009fc81f
-5 pts
https://gyazo.com/fb7b8bf2f2fd64d2d7aa211f53f7b39a
https://gyazo.com/47632dc2819812db160f4c62c8a6a03f
https://gyazo.com/0e919ff89376a341b20eac00bb928452
https://gyazo.com/802480197fc545c0d3d2af630daf6732
'''
######### functions for using the encoder, decoder
def define_models(n_input, n_output, n_units, categorical): #categorical is a boolean    
    if categorical:
        act_fxn = 'softmax'
    else:
        act_fxn = 'relu'        
   # define training encoder
    encoder_inputs = Input(shape=(None, n_input)) 
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output)) 
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)    
   #play with dense layer here
#    decoder_dense0 = Dense(n_output*5, activation = act_fxn) 
#    decoder_dense1 = Dense(n_output*3, activation = act_fxn)
    decoder_dense2 = Dense(n_output, activation = act_fxn)    
#    decoder_outputs = decoder_dense0(decoder_outputs)
#    decoder_outputs = decoder_dense1(decoder_outputs)   
    decoder_outputs = decoder_dense2(decoder_outputs)    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,)) 
    decoder_state_input_c = Input(shape=(n_units,)) 
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]    
#    decoder_outputs = decoder_dense0(decoder_outputs)
#    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)    
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    #returns train, inference_encoder and inference_decoder models
    return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char    
        yhat, h, c = infdec.predict([target_seq] + state)    
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return np.array(output)

# decode a one hot encoded string
def one(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]
###############################################################################
'''
set up problem variables 
'''
n_samples = 10000
categorical = False
n_points = 5 #points to generate 
#cardinality: points can range from 0 to c; i.e. c=1920 in the case of a 1080x1920 monitor
c = 300 #only matters if data is categorical
###############################################################################
# generate training dataset
n_features = c
if not categorical:
    c = 1 
    n_features = 1
data = np.random.random((n_samples,2,2))
x_data = data[:,:,0] # in the format [xstart, xend]
y_data = data[:,:,1] # in the format [ys, ye]
#generate control points for bezier curve by taking point 1/4 way from start to end and rotate 
theta = np.pi/4 
rot_mat = np.array( [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #ccw
#rot_mat = np.array( [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) #cw
control_points = np.array([generate_control(d[0],d[1], rot_mat) for d in data])

x_teacher = []
y_teacher =[]
x_target =[]
y_target = []

for i in range(len(data)):
    x_tea, y_tea, x_tar, y_tar = bezier_lstm_separate(
            data[i][0], control_points[i], data[i][1], 
            n_points, n_features, n_features, categorical)
    x_teacher.append(x_tea)
    y_teacher.append(y_tea)
    x_target.append(x_tar)
    y_target.append(y_tar) 

#    plt.figure() #plot to see desired result
#    plt.ylim(0,c)
#    plt.xlim(0,c)
#    if categorical:
#        plt.plot(one(x_tar),one(y_tar))
#    else:
#        plt.plot(x_tar,y_tar)    
           
if categorical:
    x_data = np.round(x_data*(c-1)) #subtract 1 bc points start at 0 (ie 0,1,2...9)
    y_data = np.round(y_data*(c-1))    
    x_data= to_categorical(x_data, c)
    y_data= to_categorical(y_data, c)
else:
    x_data = x_data.reshape(n_samples,2,1)
    y_data = y_data.reshape(n_samples,2,1)
x_teacher = np.array(x_teacher)
y_teacher = np.array(y_teacher)
x_target = np.array(x_target)
y_target =  np.array(y_target)
#print(x_data.shape, y_data.shape) 
#print(x_teacher.shape, y_teacher.shape, x_target.shape, y_target.shape)
if not categorical:
    d1,d2 = x_teacher.shape
    x_teacher = x_teacher.reshape(d1,d2,1)
    y_teacher = y_teacher.reshape(d1,d2,1)
    x_target = x_target.reshape(d1,d2,1)
    y_target = y_target.reshape(d1,d2,1)
#print(x_teacher.shape, y_teacher.shape, x_target.shape, y_target.shape)

###############################################################################
# define and train model 
train_x, infenc_x, infdec_x = define_models(n_features, n_features, 128, categorical) 
train_y, infenc_y, infdec_y = define_models(n_features, n_features, 128, categorical) 
if categorical:
    ls_fxn = 'categorical_crossentropy'
else:
    ls_fxn = 'mse'
    
train_x.compile(optimizer='adam', loss = ls_fxn)
train_y.compile(optimizer='adam', loss = ls_fxn) 
#overfitting not a problem with one epoch and a lot of data
train_x.fit([x_data, x_teacher], x_target, epochs=1)
train_y.fit([y_data, y_teacher], y_target, epochs=1)
###############################################################################
#test model
for i in range(10,20):  
    plt.figure()    
    if categorical:
        plt.ylim(-1,c+1)
        plt.xlim(-1,c+1)
        des_x = one(x_target[i])
        des_y = one(y_target[i])
    else:
        plt.ylim(-.1,1.1)
        plt.xlim(-.1,1.1)
        des_x = x_target[i]  
        des_y = y_target[i]
        
    res_x = predict_sequence(infenc_x, infdec_x, x_data[i].reshape(1,2,c), n_points, n_features)    
    res_y = predict_sequence(infenc_x, infdec_x, y_data[i].reshape(1,2,c), n_points, n_features)
    if categorical:
        res_x = one(res_x)
        res_y = one(res_y)
    plt.scatter(des_x,des_y)
    plt.scatter(res_x,res_y, c = 'r')

#plt.plot(history.history['loss'])
#plt.show()