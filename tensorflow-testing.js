import * as tf from "@tensorflow/tfjs-node";

const trainingInput = [[0, 0], [1, 0], [0, 1], [1, 1]];
const trainingInputTensor = tf.tensor(trainingInput);

const trainingOutput = [[0], [1], [1], [0]]
const trainingOutputTensor = tf.tensor(trainingOutput);

console.log(trainingInputTensor.print())
console.log(tf.tensor(trainingInput).print())

const testInput = [[1, 1], [0, 1]];
const testInputTensor = tf.tensor(testInput);

const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
model.compile({
    optimizer: tf.train.sgd(0.5),
    loss: 'binaryCrossentropy'
});

await model.fit(trainingInputTensor, trainingOutputTensor, {
    epochs: 10000,
    shuffle: false,
    verbose: false,
    callbacks: {
        onEpochEnd: async (epoch, { loss }) => {
            
            //await tf.nextFrame();
        }
    }
})

const output = model.predict(testInputTensor).arraySync();

console.log(output)

for (let i = 0; i < model.getWeights().length; i++) {
    console.log(model.getWeights()[i].dataSync());
}

await model.save('file://modelosalvo')