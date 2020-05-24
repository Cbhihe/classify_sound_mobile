# classify_sound_mobile

## Build a custom audio classifier to be trained using TensorFlow.js.

Source:
https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab/index.html#0

All that follows can be done either on a laptop or on a smartphone and was
adapated from Google codelab.

You will use transfer learning to make a model that classifies short sounds
with relatively little training data. You will be using a pre-trained model
for speech command recognition. You will train a new model on top of this
model to recognize your own custom sound classes.  We test the transferred model
by making sounds and using it to control a slider in the browser.

The Speech Command Recognizer is a JavaScript module that enables recognition
of spoken commands comprised of simple isolated English words from a small
vocabulary. The default vocabulary includes the following words:
 digits from "zero" to "nine",
 "up", "down",
 "left", "right",
 "go", "stop",
 "yes", "no",
as well as the additional categories of "unknown word" and "background noise".

It uses the web browser's WebAudio API. It is built on top of TensorFlow.js
and can perform inference and transfer learning entirely in the browser,
using WebGL GPU acceleration.

First, we load and run a pre-trained model that can recognize 20 speech
commands. Then using your microphone, you will build and train a simple neural
network that recognizes your sounds and makes the slider go left or right.

## Objective
 - load a pre-trained speech command recognition model
 - make real-time predictions using the microphone
 - train and use a custom audio recognition model using the
   browser microphone

## To know more
 - about the 'Speech Command Recognizer', visit:
   https://github.com/tensorflow/tfjs-models/tree/master/speech-commands
 - about the theory,  read the article by P. Warden (2018): "Speech commands: A
   dataset for limited-vocabulary speech recognition"s, located at:
        https://arxiv.org/pdf/1804.03209.pdf
 - about the TensorFlow Speech Commands Dataset, used to train the under-
   lying deep NN, visit:
        https://www.tensorflow.org/tutorials/sequences/audio_recognition
 - consider watching a video (as a refresher) by 3blue1brown ats:
        https://www.youtube.com/watch?v=aircAruvnKk
   or this other video on Deep Learning in Javascript by Ashi Krishnan, at:
        https://www.youtube.com/watch?v=SV-cgdobtTA
## Trials:
To run the webpage, simply open 'index.html' in a web browser. If you use
the cloud console, simply refresh the preview page. If you work from a
local file, to access the microphone you will have to start a web server
and use http://localhost:port/.
To start a simple webserver on port 8000, issue the following cmd in terminal:
   > /usr/bin/python3 -m SimpleHTTPServer

Downloading the model may take a few seconds.
As soon as the model loads, you should see a word at the top of the page.
The model was trained to recognize the numbers 0 through 9 and a few additional
commands such as "left", "right", "yes", "no", etc.
Speak one of those words. Does it get your word correctly? Play with the
'probabilityThreshold' which controls how often the model fires -- 0.75 means
that the model will fire when it is more than 75% confident that it hears a
given word.

To collect examples for each command, make a consistent sound repeatedly (or
continuously) while pressing and holding each button for 3-4 seconds. You
should collect ~150 examples for each label. For example, snap your fingers
for "Left", whistle for "Right", and alternate between silence and talk for
"Noise".
