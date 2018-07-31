'''
some of the functions I used to generate data sets
'''

import numpy as np

def constAccel(xs,xe,n):
#n is size of  output
#equiv to n-1 timesteps 
#returns array of coords assuming constant accel 

#d= v*t+1/2*a*t^2 kinematic equation 
    a= 2*(xe-xs)/(n-1)**2
    x=[]
    for i in range(n):
        x.append(xs+.5*a*i**2)
    return np.array(x)

def linspace(p1,p2,n):
    if n<2:
        print('invalid size')
        return [-1]        
    space= (p2-p1)/(n-1)
    return np.array([p1+i*space for i in range(n)])
#######################################################################
##section on Brezier curves
#generate control point for creation of brezier curve
def generate_control(start,end, rot):
    x = (end[0]-start[0])/4 
    y = (end[1]-start[1])/4 
    p = np.array([x,y])    
    return np.matmul(rot,p)+start

def bezierXYSeparate(p0,p1,p2,n):
#returns arry of (x,y) coords res= [(xs,ys),(x1,y1)...(xe,ye)]
#p0 is start point, p2 is endpoint, p1 is the 'control'
#n is size of  output
#equiv to n-1 timesteps 
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    timestep=1/(n-1)
    X=[]
    Y=[]
    for i in range(n):
        t=i*timestep
        x= (1-t)**2*x0+2*(1-t)*t*x1+t**2*x2
        y= (1-t)**2*y0+2*(1-t)*t*y1+t**2*y2
        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)

def bezierZipped(p0,p1,p2,n): #bezierXYSeparate, but output is coordinates
    x0,y0 = p0
    x1,y1 = p1
    x2,y2 = p2
    timestep=1/(n-1)
    X=[]
    Y=[]
    for i in range(n):
        t=i*timestep
        x= (1-t)**2*x0+2*(1-t)*t*x1+t**2*x2
        y= (1-t)**2*y0+2*(1-t)*t*y1+t**2*y2
        X.append(x)
        Y.append(y)
    
    return np.array(list(zip(X,Y)))
        

#bezierZipped but in categorical format
#assumes points entered are between 0 and 1, which will then be scaled by x_dim and y_dim
def bezier_lstm_zipped(p0,p1,p2,n, x_dim, y_dim, categorical = True): #point input is from 0 to 1    
    x0,y0 = p0 #assume that given endpoints are within dimensions
    x1,y1 = p1 
    x1 = _force(x1,0,1)
    y1 = _force(y1,0,1)
    x2,y2 = p2
    timestep=1/(n-1)
    X = []
    Y = []
    for i in range(n):
        t=i*timestep
        x= (1-t)**2*x0+2*(1-t)*t*x1+t**2*x2
        y= (1-t)**2*y0+2*(1-t)*t*y1+t**2*y2
        if categorical:
            X.append(round(x*(x_dim-1))) #subtract 1 bc points start at 0 (ie 0,1,2...9)
            Y.append(round(y*(y_dim-1)))       
        else:
            X.append(x)
            Y.append(y)                   
    X_teacher = [0] + X[:-1]
    Y_teacher = [0] + Y[:-1]
    
    if not categorical:      
        return np.array( list(zip(X_teacher,Y_teacher))), np.array(list(zip(X,Y)))
    
    X = to_categorical(X, num_classes = max(x_dim,y_dim))   
    Y = to_categorical(Y, num_classes = max(x_dim,y_dim)) 
    X_teacher = to_categorical(X_teacher, num_classes = max(x_dim,y_dim))   
    Y_teacher = to_categorical(Y_teacher, num_classes = max(x_dim,y_dim)) 

    return np.array( list(zip(X_teacher,Y_teacher))), np.array(list(zip(X,Y)))
 
    
def bezier_lstm_separate(p0,p1,p2,n, x_dim, y_dim, categorical = True): #point input is from 0 to 1    
    x0,y0 = p0 #assume that given endpoints are within dimensions
    x1,y1 = p1 
    x1 = _force(x1,0,1)
    y1 = _force(y1,0,1)
    x2,y2 = p2
    timestep=1/(n-1)
    X = []
    Y = []
    for i in range(n):
        t=i*timestep
        x= (1-t)**2*x0+2*(1-t)*t*x1+t**2*x2
        y= (1-t)**2*y0+2*(1-t)*t*y1+t**2*y2
        if categorical:
            X.append(round(x*(x_dim-1))) #subtract 1 bc points start at 0 (ie 0,1,2...9)
            Y.append(round(y*(y_dim-1)))       
        else:
            X.append(x)
            Y.append(y)          
    X_teacher = [0] + X[:-1]
    Y_teacher = [0] + Y[:-1]   
    
    if not categorical:
        return np.array(X_teacher), np.array(Y_teacher), np.array(X), np.array(Y)
    
    X = to_categorical(X, num_classes = max(x_dim,y_dim))   
    Y = to_categorical(Y, num_classes = max(x_dim,y_dim)) 
    X_teacher = to_categorical(X_teacher, num_classes = max(x_dim,y_dim))   
    Y_teacher = to_categorical(Y_teacher, num_classes = max(x_dim,y_dim)) 
    #return np.array( [list(zip(X_teacher,Y_teacher)),list(zip(X,Y))])
    return np.array(X_teacher), np.array(Y_teacher), np.array(X), np.array(Y)

def _force(num, lower, upper): #forces a number between bounds
    num = min(num, upper)
    num = max(num, lower)
    return num
#######################################################################


#for use in encoder/decoder 
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences as pad
#generates sequences like: [2,3,4] and [8,7,6,5,4]
def get_sequence(rangeOfSequence, mode): 
    #modes available (1,2,3,4)
    #1 categorical input, categorical output
    #2 categorical input, continous output
    #3 continous input, categorical output
    #4 continous input, continous output
    c = rangeOfSequence
  
    data = [[i,j] for i in range(1,c+1) for j in range(i,c+1) ] 
    data2 = [[j,i] for i in range(1,c+1) for j in range(i,c+1) ] 
    #the -1 serves as an 'end of sentence' indicator
    target_in = [ [[0]] + [  [k] for k in range(d[0],d[1]+1)] + [[-1]] for d in data]  #teacher forcing
    target_out = [[ [k] for k in range(d[0],d[1]+1)] + [[-1]]  + [[0]]  for d in data] #output     
    target_in2 = [ [[0]] + [  [k] for k in range(d[0],d[1]-1,-1)] + [[-1]] for d in data2]  #for the other way around
    target_out2 = [[ [k] for k in range(d[0],d[1]-1,-1)] + [[-1]]  + [[0]]  for d in data2]
    
    #combine
    target_in = target_in + target_in2
    target_out = target_out + target_out2
    
    target_in= pad(target_in, padding = 'post')
    target_out= pad(target_out, padding = 'post')
    
    data = [[[i],[j]] for i in range(1,c+1) for j in range(i,c+1) ]
    data2 = [[[j],[i]] for i in range(1,c+1) for j in range(i,c+1) ]
    data = data + data2
    
    data = np.array(data, dtype=float)
    target_out = np.array(target_out,dtype=float)
    target_in = np.array(target_in,dtype=float)    
    if mode == 4:
        return data, target_in, target_out
    
    dim1 = target_in.shape[0]
    dim2 = target_in.shape[1]    
    d = to_categorical([data], num_classes=c+2)
    d = d.reshape(dim1,2,c+2)    
    if mode == 2:
        return d, target_in, target_out
    
    t_in = to_categorical([target_in], num_classes=c+2)   
    t_in = t_in.reshape(dim1,dim2,c+2)
    
    t_out = to_categorical([target_out], num_classes=c+2)   
    t_out = t_out.reshape(dim1,dim2,c+2)    
    if mode == 3:
        return data, t_in, t_out
    
    return d, t_in, t_out #mode 1



