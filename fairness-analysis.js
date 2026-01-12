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
import {
  getAgeSpecificThreshold,
  applyRuleBasedAdjustments
} from './improve-model.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Load model from filesystem
 */
async function loadModel(modelDir) {
  const modelJsonPath = path.join(modelDir, 'model.json');
  const weightsPath = path.join(modelDir, 'weights.bin');

  const modelData = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
  const model = await tf.models.modelFromJSON(modelData.modelTopology);

  const weightsData = fs.readFileSync(weightsPath);
  const weightSpecs = modelData.weightsManifest[0].weights;

  let offset = 0;
  const weightValues = weightSpecs.map(spec => {
    const size = spec.shape.reduce((a, b) => a * b, 1);
    const bytes = size * 4;
    const buffer = weightsData.slice(offset, offset + bytes);
    offset += bytes;
    const floatArray = new Float32Array(buffer.buffer, buffer.byteOffset, size);
    return tf.tensor(Array.from(floatArray), spec.shape, spec.dtype);
  });

  model.setWeights(weightValues);
  return model;
}

/**
 * Calculate all metrics for a set of predictions
 */
function calculateMetrics(yTrue, yPred, yPredProbs) {
  let tp = 0, fp = 0, tn = 0, fn = 0;

  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 0 && yPred[i] === 0) tn++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
  }

  const total = tp + tn + fp + fn;
  const accuracy = (tp + tn) / total;
  const precision = tp / (tp + fp) || 0;
  const recall = tp / (tp + fn) || 0;
  const f1Score = 2 * (precision * recall) / (precision + recall) || 0;
  const specificity = tn / (tn + fp) || 0;
  const fpr = fp / (fp + tn) || 0; // False positive rate
  const fnr = fn / (fn + tp) || 0; // False negative rate

  // Calculate ROC-AUC
  const rocAuc = calculateROCAUC(yTrue, yPredProbs);

  // Positive prediction rate (for demographic parity)
  const ppr = (tp + fp) / total;

  return {
    sampleSize: total,
    accuracy,
    precision,
    recall,
    f1Score,
    specificity,
    fpr,
    fnr,
    rocAuc,
    ppr, // Positive prediction rate
    confusionMatrix: { tp, fp, tn, fn }
  };
}

/**
 * Calculate ROC-AUC score
 */
function calculateROCAUC(yTrue, yPredProbs) {
  const pairs = yTrue.map((label, i) => ({
    prob: yPredProbs[i],
    label: label
  }));

  pairs.sort((a, b) => b.prob - a.prob);

  let auc = 0;
  let tpCount = 0;
  const totalPositive = yTrue.filter(y => y === 1).length;
  const totalNegative = yTrue.length - totalPositive;

  if (totalPositive === 0 || totalNegative === 0) {
    return 0;
  }

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
 * Train/validation/test split (same as training and evaluation)
 */
function trainValTestSplit(features, labels, processedData, trainRatio = 0.7, valRatio = 0.15) {
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
  const shuffledData = indices.map(i => processedData[i]);

  return {
    testFeatures: shuffledFeatures.slice(trainSize + valSize),
    testLabels: shuffledLabels.slice(trainSize + valSize),
    testData: shuffledData.slice(trainSize + valSize)
  };
}

/**
 * Analyze performance by demographic subgroups
 * Now accepts pre-calculated binary predictions to support age-specific thresholds
 */
function analyzeSubgroups(testData, testLabels, predictions, predBinaryInput = null) {
  // Use provided binary predictions (with age-specific thresholds) or calculate with fixed threshold
  const predBinary = predBinaryInput || predictions.map(p => p >= 0.5 ? 1 : 0);

  // Group by age
  const ageGroups = {};
  const genderGroups = {};

  testData.forEach((row, idx) => {
    const age = row.age;
    const gender = row.gender;
    const trueLabel = testLabels[idx];
    const predProb = predictions[idx];
    const predLabel = predBinary[idx];

    // Age groups
    if (!ageGroups[age]) {
      ageGroups[age] = { trueLabels: [], predProbs: [], predLabels: [] };
    }
    ageGroups[age].trueLabels.push(trueLabel);
    ageGroups[age].predProbs.push(predProb);
    ageGroups[age].predLabels.push(predLabel);

    // Gender groups
    if (!genderGroups[gender]) {
      genderGroups[gender] = { trueLabels: [], predProbs: [], predLabels: [] };
    }
    genderGroups[gender].trueLabels.push(trueLabel);
    genderGroups[gender].predProbs.push(predProb);
    genderGroups[gender].predLabels.push(predLabel);
  });

  // Calculate metrics for each subgroup
  const ageMetrics = {};
  for (const [age, data] of Object.entries(ageGroups)) {
    ageMetrics[age] = calculateMetrics(data.trueLabels, data.predLabels, data.predProbs);
  }

  const genderMetrics = {};
  for (const [gender, data] of Object.entries(genderGroups)) {
    genderMetrics[gender] = calculateMetrics(data.trueLabels, data.predLabels, data.predProbs);
  }

  return { ageMetrics, genderMetrics };
}

/**
 * Calculate fairness metrics
 */
function calculateFairnessMetrics(subgroupMetrics) {
  const groups = Object.keys(subgroupMetrics);
  if (groups.length === 0) return {};

  // Demographic Parity: difference in positive prediction rates
  const pprs = groups.map(g => subgroupMetrics[g].ppr);
  const demographicParityDiff = Math.max(...pprs) - Math.min(...pprs);

  // Equal Opportunity: difference in true positive rates (recall)
  const tprs = groups.map(g => subgroupMetrics[g].recall);
  const equalOpportunityDiff = Math.max(...tprs) - Math.min(...tprs);

  // Equalized Odds: difference in both TPR and FPR
  const fprs = groups.map(g => subgroupMetrics[g].fpr);
  const equalizedOddsDiff = Math.max(
    Math.max(...tprs) - Math.min(...tprs),
    Math.max(...fprs) - Math.min(...fprs)
  );

  // Predictive Parity: difference in precision
  const precisions = groups.map(g => subgroupMetrics[g].precision);
  const predictiveParityDiff = Math.max(...precisions) - Math.min(...precisions);

  return {
    demographicParity: {
      difference: demographicParityDiff,
      interpretation: demographicParityDiff < 0.1 ? 'GOOD' : demographicParityDiff < 0.2 ? 'MODERATE' : 'POOR',
      description: 'Difference in positive prediction rates across groups'
    },
    equalOpportunity: {
      difference: equalOpportunityDiff,
      interpretation: equalOpportunityDiff < 0.1 ? 'GOOD' : equalOpportunityDiff < 0.2 ? 'MODERATE' : 'POOR',
      description: 'Difference in true positive rates (recall) across groups'
    },
    equalizedOdds: {
      difference: equalizedOddsDiff,
      interpretation: equalizedOddsDiff < 0.1 ? 'GOOD' : equalizedOddsDiff < 0.2 ? 'MODERATE' : 'POOR',
      description: 'Maximum difference in TPR and FPR across groups'
    },
    predictiveParity: {
      difference: predictiveParityDiff,
      interpretation: predictiveParityDiff < 0.1 ? 'GOOD' : predictiveParityDiff < 0.2 ? 'MODERATE' : 'POOR',
      description: 'Difference in precision across groups'
    }
  };
}

/**
 * Test robustness with noisy inputs
 */
async function testRobustness(model, testData, metadata) {
  console.log('Testing robustness with noisy inputs...\n');

  const results = {
    missingValues: {},
    outliers: {},
    boundaryValues: {}
  };

  // Test 1: Missing values (already handled by preprocessing, but test edge cases)
  console.log('1. Testing missing value handling...');
  const sampleWithMissing = {
    ...testData[0],
    time_in_hospital: '?', // Missing value
    num_lab_procedures: '',
    readmitted: 'NO'
  };

  try {
    const { data: processed } = preprocessData([sampleWithMissing], metadata);
    const { features } = extractFeaturesAndLabels(processed);
    const tensor = tf.tensor2d(features);
    const pred = model.predict(tensor);
    const prob = pred.dataSync()[0];
    tensor.dispose();
    pred.dispose();
    results.missingValues.handled = true;
    results.missingValues.probability = prob;
  } catch (error) {
    results.missingValues.handled = false;
    results.missingValues.error = error.message;
  }

  // Test 2: Extreme outliers
  console.log('2. Testing outlier handling...');
  const sampleWithOutliers = {
    ...testData[0],
    time_in_hospital: '999', // Extreme value
    num_lab_procedures: '10000',
    num_medications: '500',
    readmitted: 'NO'
  };

  try {
    const { data: processed } = preprocessData([sampleWithOutliers], metadata);
    const { features } = extractFeaturesAndLabels(processed);
    const tensor = tf.tensor2d(features);
    const pred = model.predict(tensor);
    const prob = pred.dataSync()[0];
    tensor.dispose();
    pred.dispose();
    results.outliers.handled = true;
    results.outliers.probability = prob;
    results.outliers.bounded = prob >= 0 && prob <= 1;
  } catch (error) {
    results.outliers.handled = false;
    results.outliers.error = error.message;
  }

  // Test 3: Boundary values
  console.log('3. Testing boundary value handling...');
  const sampleWithBoundaries = {
    time_in_hospital: '0',
    num_lab_procedures: '0',
    num_medications: '0',
    num_procedures: '0',
    number_inpatient: '0',
    number_outpatient: '0',
    number_emergency: '0',
    number_diagnoses: '0',
    age: '[0-10)',
    gender: 'Unknown/Invalid',
    readmitted: 'NO'
  };

  try {
    const { data: processed } = preprocessData([sampleWithBoundaries], metadata);
    const { features } = extractFeaturesAndLabels(processed);
    const tensor = tf.tensor2d(features);
    const pred = model.predict(tensor);
    const prob = pred.dataSync()[0];
    tensor.dispose();
    pred.dispose();
    results.boundaryValues.handled = true;
    results.boundaryValues.probability = prob;
  } catch (error) {
    results.boundaryValues.handled = false;
    results.boundaryValues.error = error.message;
  }

  return results;
}

/**
 * Generate text report
 */
function generateTextReport(overallMetrics, ageMetrics, genderMetrics, ageFairness, genderFairness, robustness) {
  let report = '';

  report += '='.repeat(80) + '\n';
  report += '                    FAIRNESS AND ROBUSTNESS ANALYSIS REPORT\n';
  report += '                    Patient Readmission Prediction Model\n';
  report += '='.repeat(80) + '\n\n';

  // Overall Performance
  report += '1. OVERALL MODEL PERFORMANCE\n';
  report += '-'.repeat(80) + '\n';
  report += `Sample Size: ${overallMetrics.sampleSize}\n`;
  report += `Accuracy: ${(overallMetrics.accuracy * 100).toFixed(2)}%\n`;
  report += `Precision: ${(overallMetrics.precision * 100).toFixed(2)}%\n`;
  report += `Recall: ${(overallMetrics.recall * 100).toFixed(2)}%\n`;
  report += `F1 Score: ${(overallMetrics.f1Score * 100).toFixed(2)}%\n`;
  report += `ROC-AUC: ${overallMetrics.rocAuc.toFixed(4)}\n\n`;

  // Subgroup Analysis - Age
  report += '2. SUBGROUP ANALYSIS: AGE GROUPS\n';
  report += '-'.repeat(80) + '\n';
  const ageGroupsSorted = Object.keys(ageMetrics).sort();
  for (const age of ageGroupsSorted) {
    const metrics = ageMetrics[age];
    report += `\nAge Group: ${age}\n`;
    report += `  Sample Size: ${metrics.sampleSize}\n`;
    report += `  Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%\n`;
    report += `  Precision: ${(metrics.precision * 100).toFixed(2)}%\n`;
    report += `  Recall: ${(metrics.recall * 100).toFixed(2)}%\n`;
    report += `  F1 Score: ${(metrics.f1Score * 100).toFixed(2)}%\n`;
    report += `  ROC-AUC: ${metrics.rocAuc.toFixed(4)}\n`;
    report += `  Positive Prediction Rate: ${(metrics.ppr * 100).toFixed(2)}%\n`;
  }

  // Subgroup Analysis - Gender
  report += '\n\n3. SUBGROUP ANALYSIS: GENDER\n';
  report += '-'.repeat(80) + '\n';
  for (const [gender, metrics] of Object.entries(genderMetrics)) {
    const genderLabel = gender === '0' ? 'Female' : gender === '1' ? 'Male' : 'Unknown/Invalid';
    report += `\nGender: ${genderLabel}\n`;
    report += `  Sample Size: ${metrics.sampleSize}\n`;
    report += `  Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%\n`;
    report += `  Precision: ${(metrics.precision * 100).toFixed(2)}%\n`;
    report += `  Recall: ${(metrics.recall * 100).toFixed(2)}%\n`;
    report += `  F1 Score: ${(metrics.f1Score * 100).toFixed(2)}%\n`;
    report += `  ROC-AUC: ${metrics.rocAuc.toFixed(4)}\n`;
    report += `  Positive Prediction Rate: ${(metrics.ppr * 100).toFixed(2)}%\n`;
  }

  // Fairness Metrics - Age
  report += '\n\n4. FAIRNESS METRICS: AGE GROUPS\n';
  report += '-'.repeat(80) + '\n';
  for (const [metric, data] of Object.entries(ageFairness)) {
    report += `\n${metric}:\n`;
    report += `  Difference: ${(data.difference * 100).toFixed(2)}%\n`;
    report += `  Assessment: ${data.interpretation}\n`;
    report += `  Description: ${data.description}\n`;
  }

  // Fairness Metrics - Gender
  report += '\n\n5. FAIRNESS METRICS: GENDER\n';
  report += '-'.repeat(80) + '\n';
  for (const [metric, data] of Object.entries(genderFairness)) {
    report += `\n${metric}:\n`;
    report += `  Difference: ${(data.difference * 100).toFixed(2)}%\n`;
    report += `  Assessment: ${data.interpretation}\n`;
    report += `  Description: ${data.description}\n`;
  }

  // Robustness Tests
  report += '\n\n6. ROBUSTNESS TESTS\n';
  report += '-'.repeat(80) + '\n';
  report += '\nMissing Values:\n';
  report += `  Handled: ${robustness.missingValues.handled ? 'YES' : 'NO'}\n`;
  if (robustness.missingValues.handled) {
    report += `  Prediction Probability: ${(robustness.missingValues.probability * 100).toFixed(2)}%\n`;
  }

  report += '\nOutliers:\n';
  report += `  Handled: ${robustness.outliers.handled ? 'YES' : 'NO'}\n`;
  if (robustness.outliers.handled) {
    report += `  Prediction Probability: ${(robustness.outliers.probability * 100).toFixed(2)}%\n`;
    report += `  Bounded (0-1): ${robustness.outliers.bounded ? 'YES' : 'NO'}\n`;
  }

  report += '\nBoundary Values:\n';
  report += `  Handled: ${robustness.boundaryValues.handled ? 'YES' : 'NO'}\n`;
  if (robustness.boundaryValues.handled) {
    report += `  Prediction Probability: ${(robustness.boundaryValues.probability * 100).toFixed(2)}%\n`;
  }

  // Summary and Recommendations
  report += '\n\n7. SUMMARY AND RECOMMENDATIONS\n';
  report += '-'.repeat(80) + '\n';
  report += '\nFairness Assessment:\n';

  const ageFairnessIssues = Object.values(ageFairness).filter(m => m.interpretation === 'POOR').length;
  const genderFairnessIssues = Object.values(genderFairness).filter(m => m.interpretation === 'POOR').length;

  if (ageFairnessIssues === 0 && genderFairnessIssues === 0) {
    report += '  Overall fairness: GOOD - No major fairness concerns detected\n';
  } else {
    report += `  Overall fairness: NEEDS ATTENTION - ${ageFairnessIssues + genderFairnessIssues} fairness metrics show poor performance\n`;
    report += '\n  Recommendations:\n';
    if (ageFairnessIssues > 0) {
      report += '  - Consider rebalancing training data across age groups\n';
      report += '  - Apply fairness constraints during training\n';
    }
    if (genderFairnessIssues > 0) {
      report += '  - Consider rebalancing training data across gender groups\n';
      report += '  - Review for potential gender bias in features\n';
    }
  }

  report += '\nRobustness Assessment:\n';
  const robustnessTests = [
    robustness.missingValues.handled,
    robustness.outliers.handled,
    robustness.boundaryValues.handled
  ];
  const robustnessPassed = robustnessTests.filter(Boolean).length;

  if (robustnessPassed === 3) {
    report += '  Model robustness: GOOD - All robustness tests passed\n';
  } else {
    report += `  Model robustness: NEEDS ATTENTION - ${3 - robustnessPassed} robustness tests failed\n`;
  }

  report += '\n' + '='.repeat(80) + '\n';
  report += 'Report generated: ' + new Date().toISOString() + '\n';
  report += '='.repeat(80) + '\n';

  return report;
}

/**
 * Main analysis function
 */
async function analyzeFairness() {
  console.log('=== FAIRNESS AND ROBUSTNESS ANALYSIS ===\n');

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
  const { testFeatures, testLabels, testData } = trainValTestSplit(features, labels, processedData);
  console.log(`Test set: ${testFeatures.length} samples\n`);

  // Make predictions with improvements
  console.log('Making predictions with fairness improvements...');
  const testXs = tf.tensor2d(testFeatures);
  const rawPredictions = model.predict(testXs);
  const rawPredProbs = Array.from(rawPredictions.dataSync());
  testXs.dispose();
  rawPredictions.dispose();

  // Apply improvements to each prediction
  const predProbs = [];
  const predBinary = [];

  for (let i = 0; i < rawPredProbs.length; i++) {
    const rawProb = rawPredProbs[i];
    const patientData = testData[i];

    // Apply rule-based adjustments
    const { adjustedProbability } = applyRuleBasedAdjustments(rawProb, patientData);

    // Get age-specific threshold
    const threshold = getAgeSpecificThreshold(patientData.age);

    // Store adjusted probability and binary classification
    predProbs.push(adjustedProbability);
    predBinary.push(adjustedProbability >= threshold ? 1 : 0);
  }

  console.log('Applied age-specific thresholds and rule-based adjustments');

  // Calculate overall metrics
  console.log('Calculating overall metrics...');
  const overallMetrics = calculateMetrics(testLabels, predBinary, predProbs);

  // Analyze subgroups (pass pre-calculated binary predictions with age-specific thresholds)
  console.log('Analyzing demographic subgroups...\n');
  const { ageMetrics, genderMetrics } = analyzeSubgroups(testData, testLabels, predProbs, predBinary);

  // Calculate fairness metrics
  console.log('Calculating fairness metrics...\n');
  const ageFairness = calculateFairnessMetrics(ageMetrics);
  const genderFairness = calculateFairnessMetrics(genderMetrics);

  // Test robustness
  const robustness = await testRobustness(model, testData, metadata);

  // Generate report
  console.log('Generating report...\n');
  const report = generateTextReport(
    overallMetrics,
    ageMetrics,
    genderMetrics,
    ageFairness,
    genderFairness,
    robustness
  );

  // Save report
  const reportPath = path.join(__dirname, '../docs/fairness-report.txt');
  fs.writeFileSync(reportPath, report);
  console.log(`Report saved to: ${reportPath}\n`);

  // Print report to console
  console.log(report);

  // Return structured results
  return {
    overallMetrics,
    ageMetrics,
    genderMetrics,
    ageFairness,
    genderFairness,
    robustness
  };
}

// Run analysis
analyzeFairness().catch(console.error);

export {
  calculateMetrics,
  analyzeSubgroups,
  calculateFairnessMetrics,
  testRobustness,
  analyzeFairness
};
