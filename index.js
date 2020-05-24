let recognizer;

function predictWord() {
  // Array of words that the recognizer is trained to recognize.

  const words = recognizer.wordLabels();

  recognizer.listen(({scores}) => {
    // Turn scores into a list of (score,word) pairs.
    scores = Array.from(scores).map((s, i) => ({score: s, word: words[i]}));
    // Find the most probable word.
    scores.sort((s1, s2) => s2.score - s1.score);
    document.querySelector('#console').textContent = scores[0].word;
  }, { probabilityThreshold: 0.75 });
}

// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
let examples = [];

// function called everytime any one of the 3 buttons in GUI is pressed
// - associates a label with the output of 'recognizer.listen()'.
function collect(label) {
 if (recognizer.isListening()) {
   return recognizer.stopListening();
 }
 if (label == null) {
   return;
 }
 // As 'includeSpectrogram' is true, recognizer.listen()' gives the raw 
 // spectrogram (frequency data) for 1 sec of audio, divided into 43 frames,
 // so each frame is ~23ms of audio:]
 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   // we want to use short sounds instead of words to control the slider;
   // so grab only the last 3 frames (~70ms) with 'NUM_FRAMES = 3'
   let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));

   // each training example will have 2 fields:
   //   label: 0, 1, and 2 for "Left", "Right" and "Noise" respectively.
   //   vals: 696 numbers holding the frequency information (spectrogram)
   examples.push({vals, label});
   document.querySelector('#console').textContent =
       `${examples.length} examples collected`;
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

// To avoid numerical issues, normalize data (center and scale it) to end up
// with an average=0 and a std-dev=1. Spectrogram values are usually large 
// negative numbers around -100 and deviation of 10:
function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

// define model's architecture
// The model has 4 layers:
// - a convolutional layer that processes the audio data (as a spectrogram),
// - a max pool layer,
// - a flatten layer, and
// - a dense layer that maps to the 3 actions
// The model's input shape is [NUM_FRAMES, 232, 1] where each frame is 23ms
// of audio containing 232 numbers that correspond to different frequencies
// (232 was chosen because it is the amount of frequency buckets needed to
// capture the human voice). Here we use samples that are 3 frames long
// (~70ms samples) since we make sounds instead of speaking whole words to
// control the slider.
function buildModel() {
 model = tf.sequential();

 model.add(tf.layers.depthwiseConv2d({
   depthMultiplier: 8,
   kernelSize: [NUM_FRAMES, 3],
   activation: 'relu',
   inputShape: INPUT_SHAPE
 }));

 model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
 model.add(tf.layers.flatten());
 model.add(tf.layers.dense({units: 3, activation: 'softmax'}));

 const optimizer = tf.train.adam(0.01);
 // compile the model with the Adam optimizer, common optimizer used in
 // deep learning, and for loss `categoricalCrossEntropy`, the standard loss
 // function used for classification. It measures how far the predicted
 // probabilities (one probability per class) are from having 100% probability
 // in the true class, and 0% probability for all the other classes. We also
 // provide accuracy as a metric to monitor, which will give us the percentage
 // of examples the model gets correct after each epoch of training.
 model.compile({
   optimizer,
   loss: 'categoricalCrossentropy',
   metrics: ['accuracy']
 });
}

// train model using collected data
// training goes 10 times (epochs) over the data using a batch size of 16 
// (processing 16 examples at a time) and shows the current accuracy on screen
async function train() {
 toggleButtons(false);
 const ys = tf.oneHot(examples.map(e => e.label), 3);
 const xsShape = [examples.length, ...INPUT_SHAPE];
 const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

 await model.fit(xs, ys, {
   batchSize: 16,
   epochs: 10,
   callbacks: {
     onEpochEnd: (epoch, logs) => {
       document.querySelector('#console').textContent =
           `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
     }
   }
 });
 tf.dispose([xs, ys]);
 toggleButtons(true);
}

function toggleButtons(enable) {
 document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

async function app() {
  recognizer = speechCommands.create('BROWSER_FFT');
  await recognizer.ensureModelLoaded();
  // predictword(); is removed below to make permit training the model
  // predictWord();
  buildModel();
}

async function moveSlider(labelTensor) {
 const label = (await labelTensor.data())[0];
 document.getElementById('console').textContent = label;
 if (label == 2) {
   return;
 }
 let delta = 0.1;
 const prevValue = +document.getElementById('output').value;
 document.getElementById('output').value =
     prevValue + (label === 0 ? -delta : delta);
}
// listens to the microphone and makes real time predictions.
// Code x very similar to the collect() method, which normalizes the
// raw spectrogram and drops all but the last NUM_FRAMES frames. The
// only difference is that we also call the trained model to get a
// prediction:
function listen() {
 if (recognizer.isListening()) {
   recognizer.stopListening();
   toggleButtons(true);
   document.getElementById('listen').textContent = 'Listen';
   return;
 }
 toggleButtons(false);
 document.getElementById('listen').textContent = 'Stop';
 document.getElementById('listen').disabled = false;

 recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
   const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);

   // Output of 'model.predict(input)' is a Tensor of shape [1, numClasses]
   // It represents a probability distribution over the number of classes.
   // Simply said, it is a set of confidence values for each of the possible
   // output classes. They sum to 1. The Tensor has an outer dimension of 1
   // because that is the size of the batch (a single example).
   const probs = model.predict(input);

   // Convert the probability distribution to a single integer representing
   // the most likely class, we call 'probs.argMax(1)' which returns the
   // class index with the highest probability. We pass a "1" as the axis
   // parameter because we want to compute the argMax over the last dimension,
   // numClasses.
   const predLabel = probs.argMax(1);

   // Update slider position with moveSlider()
   // - decreases slider value if label=0 ("Left"),
   // - increases it if label=1 ("Right")
   // - ignores input (does nothing) when label=2 ("Noise")
   await moveSlider(predLabel);

   // Clean up GPU memory by manually calling tf.dispose() on output Tensors.
   // The alternative to manual tf.dispose() is wrapping function calls in a
   // tf.tidy(), but this cannot be used with async functions.
   tf.dispose([input, probs, predLabel]);
 }, {
   overlapFactor: 0.999,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: true
 });
}

app();
