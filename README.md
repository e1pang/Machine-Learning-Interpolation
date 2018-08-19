# Machine-Learning-Interpolation


## Motivation
I was wondering if ML could be used to generate a path for mouse movement if the start and endpoints are given. I do not have data for mouse movement, so instead I tried to use ML to generate curves/sequences given the beginning and end of the sequence as a proof of concept.

## Training Data
In generateData.py are some of the functions used to generate the data sets. I started with simple sets and moved towards more complicated sets. The other 3 files are ready to be run with the bezier curve as the curve to be predicted. 

## What I did
- First I used a dense neural network.  For modelig the Bezier curve, it was simple and worked extremely well (when plotting the results, the model prediction overlaps with the training data). See: nn3_interp2D.py

- I then interpreted the problem as generating a sequence from input = [start,end] to output = [start, p1, p2...p_n-1, end] (seq2seq). This meant using LSTM encoder/decoder. For modeling the Bezier curve, it was more complicated, and did not work as well. See: LSTM_interp2D_x_y_sep.py , LSTM_interp2D_xyxy_mixed.py
  - On the other hand, for data sets that are truly sequences, the Dense network failed and the LSTM worked. See: get_sequence function in generate.py

## What I learned
- Categorical versus continuous 
  - Padding works for categorical data, but it does not work with for continuous data. 

  - An end marker improves results for categorical data, is meaningless for continuous data. 

    - In hindsight, the observed results on padding and end marker makes sense considering the meaning of categorical versus continuous data. 
  
- Having too few nodes/layers results in the model being unable to capture all of the data (underfitting), whereas having too many can lead to overfitting, with the additional problem of dead nodes from backpropagation (references: 1) [More layers versus more nodes](https://stats.stackexchange.com/questions/222883/why-are-neural-networks-becoming-deeper-but-not-wider) 2) [Dying node](https://www.quora.com/What-is-the-dying-ReLU-problem-in-neural-networks)).


- During training, luck with weight initialization is a factor. Sometimes the training gets stuck and is unable to improve on its loss, so the training has to be restarted.

## Resources
The LSTM code and guide to using an encoder/decoder was taken from here:

[A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)

[How to Develop an Encoder-Decoder Model for Sequence-to-Sequence Prediction in Keras](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)

Further Reading that is unrelated to the code and more about my interest in mimicing human cursor movement:

[Identifying Game Players with Mouse Biometrics](http://miro.enev.us/docs/mouse_ID.pdf)

[The detection of faked identity using unexpected questions and mouse dynamics](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177851#authcontrib)

[Predicting User Intention from Mouse Interaction](http://www.ieeeconfpublishing.org/cpir/UploadedFiles/paper%20(1).pdf)

[INVERSE BIOMETRICS FOR MOUSE DYNAMICS](https://www.isot.ece.uvic.ca/publications/behavioral-biometricsx/IJPRAI2203_P461.pdf)

[Fitt's Law](https://en.wikipedia.org/wiki/Fitts%27s_law)
