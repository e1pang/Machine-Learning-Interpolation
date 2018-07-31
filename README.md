# Machine-Learning-Interpolation


## Motivation
I was wondering if ML could be used to generate a path for mouse movement if the start and endpoints are given. I do not have data for mouse movement, so instead I tried to use ML to generate curves/sequences given the beginning and end of the sequence.


## Data
In generateData.py are some of the functions used to generate the data sets. I started with simple sets and moved towards more complicated sets. The other 3 files are ready to be run with the bezier curve as the curve to be predicted. 

## What I learned
-Padding works for categorical data, but it does not work with for continuous data. 

-An end marker improves results for categorical data, is meaningless for continuous data. 

  -In hindsight, this makes sense considering the meaning of categorical versus continuous data. 
  
-Having too few nodes/layers results in the model being unable to capture all of the data, whereas having too many nodes/layers can lead to dead nodes in output.

-During training, luck is a factor. Sometimes the training gets stuck and is unable to improve on its loss, so the training has to be restarted.


The LSTM code was taken from here:

https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/

Other interesting and possibly unrelated resources:

http://miro.enev.us/docs/mouse_ID.pdf

http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177851#authcontrib

http://www.ieeeconfpublishing.org/cpir/UploadedFiles/paper%20(1).pdf

https://en.wikipedia.org/wiki/Fitts%27s_law
