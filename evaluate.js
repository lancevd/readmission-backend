import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import {
  loadCSV,
  preprocessData,
  extractFeaturesAndLabels,
  loadMetadata
} from './preprocess.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Load model from filesystem
 */
async function loadModel(modelDir) {
  const modelJsonPath = path.join(modelDir, 'model.json');
  const weightsPath = path.join(modelDir, 'weights.bin');

  // Read model JSON
  const modelData = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));

  // Load model from JSON
  const model = await tf.models.modelFromJSON(modelData.modelTopology);

  // Load weights
  const weightsData = fs.readFileSync(weightsPath);
  const weightSpecs = modelData.weightsManifest[0].weights;

  // Parse weights from binary file
  let offset = 0;
  const weightValues = weightSpecs.map(spec => {
    const size = spec.shape.reduce((a, b) => a * b, 1);
    const bytes = size * 4; // float32 = 4 bytes
    const buffer = weightsData.slice(offset, offset + bytes);
    offset += bytes;

    const floatArray = new Float32Array(buffer.buffer, buffer.byteOffset, size);
    return tf.tensor(Array.from(floatArray), spec.shape, spec.dtype);
  });

  model.setWeights(weightValues);

  return model;
}

/**
 * Calculate all metrics
 */
function calculateMetrics(yTrue, yPred, yPredProbs) {
  let tp = 0, fp = 0, tn = 0, fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
  }

  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
  const specificity = tn / (tn + fp) || 0;

  // Calculate ROC-AUC
  const rocAuc = calculateROCAUC(yTrue, yPredProbs);

  return {
    accuracy,
    precision,
    recall,
    f1Score,
    specificity,
    rocAuc,
    confusionMatrix: { tp, fp, tn, fn }
  };
}

/**
 * Calculate ROC-AUC score
 */
function calculateROCAUC(yTrue, yPredProbs) {
  // Create pairs of (probability, true_label)
  const pairs = yTrue.map((label, i) => ({
    prob: yPredProbs[i],
    label: label
  }));

  // Sort by probability descending
  pairs.sort((a, b) => b.prob - a.prob);

  let auc = 0;
  let tpCount = 0;
  const totalPositive = yTrue.filter(y => y === 1).length;
  const totalNegative = yTrue.length - totalPositive;

  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].label === 1) {
      tpCount++;
    } else {
      auc += tpCount;
    }
  }

  return auc / (totalPositive * totalNegative);
}

/**
 * Evaluate model by threshold
 */
function evaluateByThreshold(yTrue, yPredProbs, threshold) {
  const yPred = yPredProbs.map(p => p >= threshold ? 1 : 0);
  return calculateMetrics(yTrue, yPred, yPredProbs);
}

/**
 * Find optimal threshold
 */
function findOptimalThreshold(yTrue, yPredProbs) {
  const thresholds = [];
  for (let t = 0.1; t <= 0.9; t += 0.05) {
    const metrics = evaluateByThreshold(yTrue, yPredProbs, t);
    thresholds.push({
      threshold: t,
      f1Score: metrics.f1Score,
      precision: metrics.precision,
      recall: metrics.recall
    });
  }

  // Find threshold with best F1 score
  thresholds.sort((a, b) => b.f1Score - a.f1Score);
  return thresholds[0];
}

/**
 * Train/validation/test split (same as training)
 */
function trainValTestSplit(features, labels, trainRatio = 0.7, valRatio = 0.15) {
  const totalSamples = features.length;
  const trainSize = Math.floor(totalSamples * trainRatio);
  const valSize = Math.floor(totalSamples * valRatio);

  // Shuffle with same seed for reproducibility
  const indices = Array.from({ length: totalSamples }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const shuffledFeatures = indices.map(i => features[i]);
  const shuffledLabels = indices.map(i => labels[i]);

  return {
    testFeatures: shuffledFeatures.slice(trainSize + valSize),
    testLabels: shuffledLabels.slice(trainSize + valSize)
  };
}

/**
 * Main evaluation function
 */
async function evaluate() {
  console.log('=== MODEL EVALUATION ===\n');

  // Set backend
  await tf.setBackend('cpu');
  await tf.ready();

  // Load model
  console.log('Loading model...');
  const modelDir = path.join(__dirname, 'models', 'readmission-model');
  const model = await loadModel(modelDir);
  console.log('Model loaded successfully\n');

  // Load preprocessing metadata
  const metadataPath = path.join(__dirname, 'models', 'preprocess.json');
  const metadata = loadMetadata(metadataPath);

  // Load and preprocess data
  console.log('Loading dataset...');
  const csvPath = path.join(__dirname, '../data/raw/diabetic_data.csv');
  const rawData = loadCSV(csvPath);
  const { data: processedData } = preprocessData(rawData, metadata);
  const { features, labels } = extractFeaturesAndLabels(processedData);

  // Get test set
  const { testFeatures, testLabels } = trainValTestSplit(features, labels);
  console.log(`Test set: ${testFeatures.length} samples\n`);

  // Make predictions
  console.log('Making predictions...');
  const testXs = tf.tensor2d(testFeatures);
  const predictions = model.predict(testXs);
  const predProbs = Array.from(predictions.dataSync());
  const predBinary = predProbs.map(p => p >= 0.5 ? 1 : 0);

  // Calculate metrics at default threshold (0.5)
  console.log('=== METRICS AT THRESHOLD 0.5 ===\n');
  const defaultMetrics = calculateMetrics(testLabels, predBinary, predProbs);

  console.log(`Accuracy:    ${(defaultMetrics.accuracy * 100).toFixed(2)}%`);
  console.log(`Precision:   ${(defaultMetrics.precision * 100).toFixed(2)}%`);
  console.log(`Recall:      ${(defaultMetrics.recall * 100).toFixed(2)}%`);
  console.log(`F1 Score:    ${(defaultMetrics.f1Score * 100).toFixed(2)}%`);
  console.log(`Specificity: ${(defaultMetrics.specificity * 100).toFixed(2)}%`);
  console.log(`ROC-AUC:     ${defaultMetrics.rocAuc.toFixed(4)}`);

  console.log('\nConfusion Matrix:');
  console.log(`  TP: ${defaultMetrics.confusionMatrix.tp.toString().padStart(5)}  FP: ${defaultMetrics.confusionMatrix.fp.toString().padStart(5)}`);
  console.log(`  FN: ${defaultMetrics.confusionMatrix.fn.toString().padStart(5)}  TN: ${defaultMetrics.confusionMatrix.tn.toString().padStart(5)}`);

  // Find optimal threshold
  console.log('\n=== THRESHOLD OPTIMIZATION ===\n');
  const optimal = findOptimalThreshold(testLabels, predProbs);
  console.log(`Optimal threshold: ${optimal.threshold.toFixed(2)}`);
  console.log(`F1 Score at optimal: ${(optimal.f1Score * 100).toFixed(2)}%`);
  console.log(`Precision at optimal: ${(optimal.precision * 100).toFixed(2)}%`);
  console.log(`Recall at optimal: ${(optimal.recall * 100).toFixed(2)}%`);

  // Evaluate at optimal threshold
  const optimalPredBinary = predProbs.map(p => p >= optimal.threshold ? 1 : 0);
  const optimalMetrics = calculateMetrics(testLabels, optimalPredBinary, predProbs);

  console.log('\n=== METRICS AT OPTIMAL THRESHOLD ===\n');
  console.log(`Accuracy:    ${(optimalMetrics.accuracy * 100).toFixed(2)}%`);
  console.log(`Precision:   ${(optimalMetrics.precision * 100).toFixed(2)}%`);
  console.log(`Recall:      ${(optimalMetrics.recall * 100).toFixed(2)}%`);
  console.log(`F1 Score:    ${(optimalMetrics.f1Score * 100).toFixed(2)}%`);
  console.log(`Specificity: ${(optimalMetrics.specificity * 100).toFixed(2)}%`);

  console.log('\nConfusion Matrix:');
  console.log(`  TP: ${optimalMetrics.confusionMatrix.tp.toString().padStart(5)}  FP: ${optimalMetrics.confusionMatrix.fp.toString().padStart(5)}`);
  console.log(`  FN: ${optimalMetrics.confusionMatrix.fn.toString().padStart(5)}  TN: ${optimalMetrics.confusionMatrix.tn.toString().padStart(5)}`);

  // Class distribution
  const positiveCount = testLabels.filter(l => l === 1).length;
  const negativeCount = testLabels.length - positiveCount;
  console.log('\n=== TEST SET CLASS DISTRIBUTION ===\n');
  console.log(`Positive (readmitted <30): ${positiveCount} (${((positiveCount / testLabels.length) * 100).toFixed(2)}%)`);
  console.log(`Negative (not readmitted): ${negativeCount} (${((negativeCount / testLabels.length) * 100).toFixed(2)}%)`);

  // Clean up
  testXs.dispose();
  predictions.dispose();

  console.log('\n=== EVALUATION COMPLETE ===');

  // Return results for documentation
  return {
    defaultMetrics,
    optimalThreshold: optimal.threshold,
    optimalMetrics,
    testSetSize: testLabels.length,
    positiveCount,
    negativeCount
  };
}

// Run evaluation
evaluate().catch(err => {
  console.error('Error during evaluation:', err);
  process.exit(1);
});
