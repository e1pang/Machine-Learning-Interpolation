from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from generateData import generate_control, bezierXYSeparate, bezierZipped 
'''
-use only DENSE layers to predict curve: WORKED THE BEST
-ONE model predicts BOTH X AND Y coords
    -model input: start and end coords; output: coords in path 

-just like with the lstm, sometimes the training gets stuck in a local optima,
and must be restarted

-having too many nodes/layers is not always good, can lead to dead nodes in output

-result is very good, model prediction lies on TOP of training data:
https://gyazo.com/34c83a2af1147565d6c6d4b1d5d91329
'''
  
#How many points to predict
#as n_points changes, so should the layers and nodes
n_points = 5  #the models for 5, 20, 50 are provided
n_samples = 150000

data = np.random.random((n_samples,2,2)) #startpoint- data[i][0], endpt- data[i][1]

#generate control points for bezier by taking point 1/4 way from start to end and rotate 
theta = np.pi/4
rot_mat = np.array( [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #ccw
#rot_mat = np.array( [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) #cw
control_points = np.array([generate_control(d[0],d[1], rot_mat) for d in data])

##run this to visualize the curve to model
#for i in range(len(data)): 
#    d = data[i]
#    c = control_points[i]
#    plt.figure()
#    plt.ylim(0,1)
#    plt.xlim(0,1)
##to see p0, p1, p2
##    plt.scatter([d[0][0], d[1][0]], [d[0][1],d[1][1]]) 
##    plt.scatter(c[0],c[1])
#    x,y = bezierXYSeparate(d[0], c, d[1], n_points) #def bezier(p0,p1,p2,n)
#    plt.scatter(x,y)
#    plt.show

#flatten to prepare data for training 
#evens (0,2,4....) are x coord and odds(1,3..) are the y coord
x_train = np.array([d.flatten() for d in data])
y_train = np.array([bezierZipped(data[i][0],control_points[i],data[i][1], n_points).flatten() for i in range(len(data))])

model = Sequential()
'''
##uncomment this section for n_points=5
##using 'model.add(Dense(16, input_shape = (4,)))' instead does NOT work
##having too few or too many nodes/layers is detrimental
'''
model.add(Dense(4, input_shape = (4,)))

'''
##uncomment this section for n_points=20
##to increase the number of points to model, must increase the layers/number of nodes
model.add(Dense(16, input_shape = (4,)))
model.add(Dense(100)) 
model.add(Dense(100)) 
'''

'''
##uncomment this section for n_points=50
model.add(Dense(400, input_shape = (4,)))
model.add(Dense(400)) 
model.add(Dense(400)) 
model.add(Dropout(0.1)) 
'''

model.add(Dense(n_points*2, activation='relu'))

model.compile(loss='mse', optimizer='adam')
history = model.fit(x_train, y_train, epochs=1)

#test model
x_test = np.random.random((10,2,2))
x_pred = np.array([d.flatten() for d in x_test])
control_points_test = np.array([generate_control(d[0],d[1], rot_mat) for d in x_test])

des = np.array([bezierZipped(x_test[i][0],control_points_test[i],x_test[i][1], n_points).flatten() for i in range(len(x_test))])
res = model.predict(x_pred)

for i in range(len(x_test)):
    plt.figure()
    plt.ylim(0,1)
    plt.xlim(0,1)
    
    d = des[i] #desired is what we wanted to model
    x_des = [d[i] for i in range(0,len(d),2)] 
    y_des = [d[i] for i in range(1,len(d),2)]
    plt.scatter(x_des, y_des)
    
    r = res[i] #result is what the model predicted
    x_res = [r[i] for i in range(0,len(r),2)] 
    y_res = [r[i] for i in range(1,len(r),2)]
    plt.scatter(x_res, y_res)
#plt.close('all')

#plt.plot(history.history['loss'])
#plt.show()