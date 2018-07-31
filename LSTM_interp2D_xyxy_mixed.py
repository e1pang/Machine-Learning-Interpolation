import numpy as np
from numpy import argmax
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split as tts
from generateData import bezier_lstm_zipped, generate_control

'''
-LSTM cells, encoder/decoder and teacher forcing
-ONE model predicts BOTH X AND Y coords: DID NOT WORK     
    -model input: start and end coords; output: coords in path 

-why do dense nodes succeed but LSTM nodes fail? because of their different
    ways of generating the output
        - Dense: nodes in the layers able to treat x and y-coords differently
        - LSTM: there is one encoder and one decoder for both x and y coords
        - in other words: LSTM couldn't 'split' x and y coords while Dense could

screenshots 
theta = pi/4
https://gyazo.com/a13af219d59811b2c843fc2eaadb3111
https://gyazo.com/c2a4709d75874701ad13d77d67b0d6df
https://gyazo.com/15243a0d08b71421ab4229c03cb0c229
https://gyazo.com/8fb50df45d2c181280648ec3f35db2a0
https://gyazo.com/25fe7690d7c4df6e0ac8abcec92912bf
https://gyazo.com/f4e8e0ec66c49fb795e18507009e37e9
https://gyazo.com/05623fb475ce5eccf6cc95cdea4f67b5
https://gyazo.com/b14424e716a0981d152baaea74ae61bc

theta = pi/2
https://gyazo.com/ffd98716bd2d401e20fbbc2b1abb6b68
https://gyazo.com/599ec3368af40932769b06ad13a35399
https://gyazo.com/7d432d9c5ca5ec12a21d75bd8e6e076d

can't even get straight lines: theta = 0
https://gyazo.com/0f88826b03734dad60ec85d66f176e96
https://gyazo.com/83a60cc852347b48605cab0be10dc8af
https://gyazo.com/acf07406d29b80c44c9fbd104a728737
https://gyazo.com/1ca6c30ae49f10e63c8080a0857d707b
'''

def define_models(n_input, n_output, n_units, categorical): #  categorical is T/F
    # define training encoder
    if categorical:
        act_fxn = 'softmax'
    else:
        act_fxn = 'relu'        
   
    encoder_inputs = Input(shape=(None, n_input)) #
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output)) 
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)    
   
    decoder_dense0 = Dense(n_output*5, activation = act_fxn) #play with dense layer here
    decoder_dense1 = Dense(n_output*5, activation = act_fxn)
    decoder_dense2 = Dense(n_output, activation = act_fxn)
    
    decoder_outputs = decoder_dense0(decoder_outputs)
    decoder_outputs = decoder_dense1(decoder_outputs)   
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
    
    decoder_outputs = decoder_dense0(decoder_outputs)
    decoder_outputs = decoder_dense1(decoder_outputs)
    decoder_outputs = decoder_dense2(decoder_outputs)
    
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model
# returns train, inference_encoder and inference_decoder models

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
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]
def one(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# configure problem: for simplicity, assume that the monitor is 20 x 20
c = 10 #cardinality, points can range from 0 to c; i.e. c=1920 in the case of a 1080x1920 monitor
n_features = c
n_points = 5 #points to generate (includes start and endpoint, so min is 3 to mean anything)

n_samples = 50000
categorical = True

if not categorical:
    c = 1
    n_features = 1

# generate training dataset
data = np.random.random((n_samples,2,2)) #startpoint- data[i][0], endpt- data[i][1]
#generate control points for bezier by taking point 1/4 way from start to end and rotate 
theta = 0 #np.pi/4 
rot_mat = np.array( [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #ccw
#rot_mat = np.array( [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) #cw
control_points = np.array([generate_control(d[0],d[1], rot_mat) for d in data])

#make data categorical
x2 = []
y = []
for i in range(len(data)):
    gen_x2, gen_y = bezier_lstm_zipped(data[i][0], control_points[i], data[i][1], n_points, n_features, n_features, categorical)
    x2.append(gen_x2)
    y.append(gen_y)
#    plt.figure() #plot to see teacher forcing
#    plt.ylim(0,c)
#    plt.xlim(0,c)
#    px = [z[0] for z in gen_x2]
#    py = [z[1] for z in gen_x2]
#    plt.plot(px,py)

#    plt.figure() #plot to see desired result
#    plt.ylim(0,c)
#    plt.xlim(0,c)
#    px = [one_hot_decode(z)[0] for z in gen_y]
#    py = [one_hot_decode(z)[1] for z in gen_y]
#    plt.plot(px,py)
         ##plt.close('all')
if categorical:
    data = np.round(data*(c-1)) #subtract 1 bc points start at 0 (ie 0,1,2...9)
    x1 = to_categorical(data, num_classes = c)
else:
    x1 = data
x2 = np.array(x2)
y = np.array(y)

print(x1.shape,x2.shape,y.shape) 
if categorical:
    d1,d2,d3,d4 = x1.shape
    x1r = x1.reshape(d1, d2*d3, d4) #reshape so that each timestep/point enters as [xs,ys,xe,ye]
    d1,d2,d3,d4 = x2.shape
    x2r = x2.reshape(d1, d2*d3, d4) 
    yr = y.reshape(d1, d2*d3, d4)   
else:
    d1,d2,d3 = x1.shape
    x1r = x1.reshape(d1, d2*d3, 1) #reshape so that each timestep/point enters as [xs,ys,xe,ye]
    d1,d2,d3 = x2.shape
    x2r = x2.reshape(d1, d2*d3, 1) 
    yr = y.reshape(d1, d2*d3, 1)
print(x1r.shape,x2r.shape,yr.shape)

#before reshape, a series of 5 points would be: [[33, 7], [29, 7], [25, 10], [20, 14], [13, 21]]
#after reshape:  [33, 7, 29, 7, 25, 10, 20, 14, 13, 21]
# pattern is x1,y1,x2,...,x5,y5

#X1, X1_split, X2, X2_split, y, y_split = tts(X1,X2, y, test_size=.2, random_state =4) 
# define and train model 
train2, infenc2, infdec2 = define_models(n_features, n_features, 128, categorical) 
if categorical:
    train2.compile(optimizer='adam', loss='categorical_crossentropy')
else:
    train2.compile(optimizer='adam', loss='mse') 
    
history = train2.fit([x1r, x2r], yr, epochs=1)# validation_data=([X1_split, X2_split], y_split)) 
#
##for i in range(len(data)):
for i in range(0,10):
    plt.figure()
    if categorical:
        plt.ylim(-1,c+1)
        plt.xlim(-1,c+1)
        des = one(yr[i])   
    else:
        plt.ylim(-.1,1.1)
        plt.xlim(-.1,1.1)
        des = yr[i]       
    
    x_des = [des[i] for i in range(0,len(des),2)]
    y_des = [des[i] for i in range(1,len(des),2)]
    plt.plot(x_des,y_des)    

    xx = x1r[i]
    #print(one(x1[i][0]),one(x1[i][1]),one(xx))
    #print(xx)
    res = predict_sequence(infenc2, infdec2, xx.reshape(1,4,c), n_points*2, n_features)
    if categorical:
        res = one_hot_decode(res)
    x_res = [res[i] for i in range(0,len(res),2)]
    y_res = [res[i] for i in range(1,len(res),2)]
    x_res= np.array(x_res).flatten()
    y_res= np.array(y_res).flatten()

    plt.plot(x_res,y_res, c='r')

plt.plot(history.history['loss'])
plt.show()