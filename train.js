import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import {
  loadCSV,
  preprocessData,
  extractFeaturesAndLabels,
  saveMetadata
} from './preprocess.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Split data into train, validation, and test sets
 */
function trainValTestSplit(features, labels, trainRatio = 0.7, valRatio = 0.15) {
  const totalSamples = features.length;
  const trainSize = Math.floor(totalSamples * trainRatio);
  const valSize = Math.floor(totalSamples * valRatio);

  // Shuffle data
  const indices = Array.from({ length: totalSamples }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const shuffledFeatures = indices.map(i => features[i]);
  const shuffledLabels = indices.map(i => labels[i]);

  return {
    trainFeatures: shuffledFeatures.slice(0, trainSize),
    trainLabels: shuffledLabels.slice(0, trainSize),
    valFeatures: shuffledFeatures.slice(trainSize, trainSize + valSize),
    valLabels: shuffledLabels.slice(trainSize, trainSize + valSize),
    testFeatures: shuffledFeatures.slice(trainSize + valSize),
    testLabels: shuffledLabels.slice(trainSize + valSize)
  };
}

/**
 * Calculate class weights for imbalanced dataset
 */
function calculateClassWeights(labels) {
  const totalSamples = labels.length;
  const positiveCount = labels.filter(l => l === 1).length;
  const negativeCount = totalSamples - positiveCount;

  // Weight inversely proportional to class frequency
  const classWeight = {
    0: totalSamples / (2 * negativeCount),
    1: totalSamples / (2 * positiveCount)
  };

  console.log('\nClass distribution:');
  console.log(`  Negative (0): ${negativeCount} (${((negativeCount / totalSamples) * 100).toFixed(2)}%)`);
  console.log(`  Positive (1): ${positiveCount} (${((positiveCount / totalSamples) * 100).toFixed(2)}%)`);
  console.log('\nClass weights:');
  console.log(`  Class 0: ${classWeight[0].toFixed(4)}`);
  console.log(`  Class 1: ${classWeight[1].toFixed(4)}`);

  return classWeight;
}

/**
 * Create sample weights array for training
 */
function createSampleWeights(labels, classWeight) {
  return labels.map(label => classWeight[label]);
}

/**
 * Build baseline neural network model
 */
function buildModel(inputShape) {
  const model = tf.sequential();

  // Input layer + Hidden layer 1
  model.add(tf.layers.dense({
    inputShape: [inputShape],
    units: 32,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));

  // Dropout for regularization
  model.add(tf.layers.dropout({ rate: 0.3 }));

  // Hidden layer 2
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    kernelInitializer: 'heNormal'
  }));

  // Dropout for regularization
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Output layer (binary classification)
  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  }));

  return model;
}

/**
 * Main training function
 */
async function train() {
  console.log('=== PATIENT READMISSION MODEL TRAINING ===\n');

  // Set CPU backend
  await tf.setBackend('cpu');
  await tf.ready();
  console.log(`Using TensorFlow.js backend: ${tf.getBackend()}\n`);

  // Load and preprocess data
  console.log('Loading dataset...');
  const csvPath = path.join(__dirname, '../data/raw/diabetic_data.csv');
  const rawData = loadCSV(csvPath);
  console.log(`Loaded ${rawData.length} records\n`);

  console.log('Preprocessing data...');
  const { data: processedData, metadata } = preprocessData(rawData);
  const { features, labels } = extractFeaturesAndLabels(processedData);
  console.log(`Preprocessed ${features.length} samples with ${features[0].length} features\n`);

  // Save preprocessing metadata
  const metadataPath = path.join(__dirname, 'models', 'preprocess.json');
  saveMetadata(metadata, metadataPath);

  // Split data
  console.log('Splitting data into train/val/test sets...');
  const {
    trainFeatures,
    trainLabels,
    valFeatures,
    valLabels,
    testFeatures,
    testLabels
  } = trainValTestSplit(features, labels, 0.7, 0.15);

  console.log(`  Train: ${trainFeatures.length} samples`);
  console.log(`  Validation: ${valFeatures.length} samples`);
  console.log(`  Test: ${testFeatures.length} samples\n`);

  // Calculate class weights
  const classWeight = calculateClassWeights(trainLabels);

  // Convert to tensors
  console.log('\nConverting data to tensors...');
  const trainXs = tf.tensor2d(trainFeatures);
  const trainYs = tf.tensor2d(trainLabels, [trainLabels.length, 1]);
  const valXs = tf.tensor2d(valFeatures);
  const valYs = tf.tensor2d(valLabels, [valLabels.length, 1]);

  // Build model
  console.log('Building model...');
  const model = buildModel(features[0].length);

  // Compile model
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  });

  console.log('\nModel architecture:');
  model.summary();

  // Train model
  console.log('\n\nTraining model...\n');
  const history = await model.fit(trainXs, trainYs, {
    epochs: 20,
    batchSize: 256,
    validationData: [valXs, valYs],
    classWeight: classWeight,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch + 1}/20 - ` +
          `loss: ${logs.loss.toFixed(4)}, ` +
          `acc: ${logs.acc.toFixed(4)} | ` +
          `val_loss: ${logs.val_loss.toFixed(4)}, ` +
          `val_acc: ${logs.val_acc.toFixed(4)}`
        );
      }
    },
    shuffle: true
  });

  console.log('\n\nTraining complete!');

  // Evaluate on test set
  console.log('\nEvaluating on test set...');
  const testXs = tf.tensor2d(testFeatures);
  const testYs = tf.tensor2d(testLabels, [testLabels.length, 1]);

  const testResults = model.evaluate(testXs, testYs);
  console.log(`Test loss: ${testResults[0].dataSync()[0].toFixed(4)}`);
  console.log(`Test accuracy: ${testResults[1].dataSync()[0].toFixed(4)}`);

  // Get predictions for manual metric calculation
  const predictions = model.predict(testXs);
  const predArray = Array.from(predictions.dataSync());
  const predBinary = predArray.map(p => p >= 0.5 ? 1 : 0);

  // Calculate confusion matrix
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < testLabels.length; i++) {
    if (testLabels[i] === 1 && predBinary[i] === 1) tp++;
    else if (testLabels[i] === 0 && predBinary[i] === 1) fp++;
    else if (testLabels[i] === 0 && predBinary[i] === 0) tn++;
    else if (testLabels[i] === 1 && predBinary[i] === 0) fn++;
  }

  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

  console.log(`\nConfusion Matrix:`);
  console.log(`  TP: ${tp}, FP: ${fp}`);
  console.log(`  FN: ${fn}, TN: ${tn}`);
  console.log(`\nTest precision: ${precision.toFixed(4)}`);
  console.log(`Test recall: ${recall.toFixed(4)}`);
  console.log(`Test F1 score: ${f1Score.toFixed(4)}`);

  // Save model manually to filesystem
  const modelDir = path.join(__dirname, 'models', 'readmission-model');
  console.log(`\nSaving model to ${modelDir}...`);

  try {
    // Create directory if it doesn't exist
    if (!fs.existsSync(modelDir)) {
      fs.mkdirSync(modelDir, { recursive: true });
    }

    // Get model topology and weights
    const modelJSON = model.toJSON(null, false);
    const weights = model.getWeights();

    // Create weights manifest
    const weightsManifest = [{
      paths: ['weights.bin'],
      weights: weights.map((w, i) => ({
        name: w.name || `weight_${i}`,
        shape: Array.from(w.shape),
        dtype: w.dtype
      }))
    }];

    // Save model.json
    const modelData = {
      modelTopology: modelJSON,
      weightsManifest: weightsManifest,
      format: 'layers-model',
      generatedBy: 'TensorFlow.js tfjs-layers v4.22.0',
      convertedBy: null
    };

    fs.writeFileSync(
      path.join(modelDir, 'model.json'),
      JSON.stringify(modelData, null, 2)
    );

    // Save weights to binary file
    const weightBuffers = weights.map(w => Buffer.from(w.dataSync().buffer));
    const weightsData = Buffer.concat(weightBuffers);
    fs.writeFileSync(path.join(modelDir, 'weights.bin'), weightsData);

    console.log('Model saved successfully!');
    console.log(`  - model.json: ${(fs.statSync(path.join(modelDir, 'model.json')).size / 1024).toFixed(2)} KB`);
    console.log(`  - weights.bin: ${(fs.statSync(path.join(modelDir, 'weights.bin')).size / 1024).toFixed(2)} KB`);

    // Clean up weight tensors
    weights.forEach(w => w.dispose());
  } catch (err) {
    console.error('Error saving model:', err.message);
    console.log('Model training completed but could not be saved to disk');
  }

  // Clean up tensors
  trainXs.dispose();
  trainYs.dispose();
  valXs.dispose();
  valYs.dispose();
  testXs.dispose();
  testYs.dispose();
  predictions.dispose();
  testResults.forEach(t => t.dispose());

  console.log('\n=== TRAINING COMPLETE ===');
}

// Run training
train().catch(err => {
  console.error('Error during training:', err);
  process.exit(1);
});
