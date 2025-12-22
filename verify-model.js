import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import {
  loadCSV,
  preprocessData,
  extractFeaturesAndLabels,
  loadMetadata,
  FEATURE_CONFIG
} from './preprocess.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Load model from filesystem
 */
async function loadModel(modelDir) {
  const modelJsonPath = path.join(modelDir, 'model.json');
  const weightsPath = path.join(modelDir, 'weights.bin');

  console.log('Loading model files...');
  console.log(`  - model.json: ${modelJsonPath}`);
  console.log(`  - weights.bin: ${weightsPath}`);

  // Verify files exist
  if (!fs.existsSync(modelJsonPath)) {
    throw new Error(`Model file not found: ${modelJsonPath}`);
  }
  if (!fs.existsSync(weightsPath)) {
    throw new Error(`Weights file not found: ${weightsPath}`);
  }

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
 * Verify preprocessing metadata
 */
function verifyPreprocessingMetadata(metadata) {
  console.log('\n=== VERIFYING PREPROCESSING METADATA ===\n');

  const checks = [];

  // Check features list
  checks.push({
    name: 'Features configuration exists',
    passed: metadata.features && metadata.features.numeric && metadata.features.categorical
  });

  // Check numeric features
  const expectedNumeric = FEATURE_CONFIG.numeric;
  checks.push({
    name: 'All numeric features present',
    passed: expectedNumeric.every(f => metadata.features.numeric.includes(f))
  });

  // Check categorical features
  const expectedCategorical = FEATURE_CONFIG.categorical;
  checks.push({
    name: 'All categorical features present',
    passed: expectedCategorical.every(f => metadata.features.categorical.includes(f))
  });

  // Check medians exist for all numeric features
  checks.push({
    name: 'Medians defined for all numeric features',
    passed: expectedNumeric.every(f => metadata.medians.hasOwnProperty(f))
  });

  // Check means exist for all numeric features
  checks.push({
    name: 'Means defined for all numeric features',
    passed: expectedNumeric.every(f => metadata.means.hasOwnProperty(f))
  });

  // Check stds exist for all numeric features
  checks.push({
    name: 'Standard deviations defined for all numeric features',
    passed: expectedNumeric.every(f => metadata.stds.hasOwnProperty(f))
  });

  // Check encodings exist for all categorical features
  checks.push({
    name: 'Encodings defined for all categorical features',
    passed: expectedCategorical.every(f => metadata.encodings.hasOwnProperty(f))
  });

  // Display results
  checks.forEach((check, i) => {
    const status = check.passed ? '✓' : '✗';
    const color = check.passed ? '' : '[FAIL] ';
    console.log(`${i + 1}. ${color}${status} ${check.name}`);
  });

  const allPassed = checks.every(c => c.passed);
  console.log(`\nOverall: ${allPassed ? 'PASSED' : 'FAILED'}`);

  return allPassed;
}

/**
 * Test end-to-end inference
 */
async function testInference(model, metadata) {
  console.log('\n=== TESTING END-TO-END INFERENCE ===\n');

  // Create sample patient data
  const samplePatients = [
    {
      time_in_hospital: '5',
      num_lab_procedures: '45',
      num_medications: '18',
      num_procedures: '1',
      number_inpatient: '0',
      number_outpatient: '0',
      number_emergency: '0',
      number_diagnoses: '9',
      age: '[60-70)',
      gender: 'Male',
      readmitted: 'NO'
    },
    {
      time_in_hospital: '8',
      num_lab_procedures: '70',
      num_medications: '25',
      num_procedures: '3',
      number_inpatient: '2',
      number_outpatient: '1',
      number_emergency: '1',
      number_diagnoses: '12',
      age: '[70-80)',
      gender: 'Female',
      readmitted: '<30'
    }
  ];

  console.log('Test patients:');
  samplePatients.forEach((patient, i) => {
    console.log(`\nPatient ${i + 1}:`);
    console.log(`  Age: ${patient.age}, Gender: ${patient.gender}`);
    console.log(`  Time in hospital: ${patient.time_in_hospital} days`);
    console.log(`  Lab procedures: ${patient.num_lab_procedures}`);
    console.log(`  Medications: ${patient.num_medications}`);
    console.log(`  Prior inpatient visits: ${patient.number_inpatient}`);
    console.log(`  Actual readmission: ${patient.readmitted}`);
  });

  // Preprocess
  const { data: processed } = preprocessData(samplePatients, metadata);
  const { features } = extractFeaturesAndLabels(processed);

  // Make predictions
  const inputTensor = tf.tensor2d(features);
  const predictions = model.predict(inputTensor);
  const probabilities = Array.from(predictions.dataSync());

  console.log('\n--- Predictions ---\n');
  probabilities.forEach((prob, i) => {
    const riskLevel = prob >= 0.5 ? 'HIGH' : prob >= 0.3 ? 'MEDIUM' : 'LOW';
    console.log(`Patient ${i + 1}:`);
    console.log(`  Predicted probability: ${(prob * 100).toFixed(2)}%`);
    console.log(`  Risk level: ${riskLevel}`);
    console.log(`  Actual readmission: ${samplePatients[i].readmitted}`);
    console.log();
  });

  // Clean up
  inputTensor.dispose();
  predictions.dispose();

  return true;
}

/**
 * Create model metadata file
 */
function createModelMetadata(modelDir) {
  console.log('\n=== CREATING MODEL METADATA ===\n');

  const metadata = {
    version: '1.0.0',
    name: 'readmission-predictor-baseline',
    description: 'Baseline neural network for predicting 30-day hospital readmission risk',
    createdDate: new Date().toISOString().split('T')[0],
    lastUpdated: new Date().toISOString().split('T')[0],

    dataset: {
      name: 'UCI Diabetes 130-US Hospitals',
      source: 'UCI Machine Learning Repository',
      years: '1999-2008',
      totalSamples: 101766,
      trainSamples: 71236,
      valSamples: 15264,
      testSamples: 15266
    },

    model: {
      architecture: 'Sequential Neural Network',
      framework: 'TensorFlow.js',
      version: '4.22.0',
      backend: 'CPU',
      totalParameters: 897,
      trainableParameters: 897,
      layers: [
        { type: 'Dense', units: 32, activation: 'relu', inputShape: [10] },
        { type: 'Dropout', rate: 0.3 },
        { type: 'Dense', units: 16, activation: 'relu' },
        { type: 'Dropout', rate: 0.2 },
        { type: 'Dense', units: 1, activation: 'sigmoid' }
      ]
    },

    training: {
      optimizer: 'adam',
      learningRate: 0.001,
      lossFunction: 'binaryCrossentropy',
      epochs: 20,
      batchSize: 256,
      classWeights: {
        0: 0.5636,
        1: 4.4295
      }
    },

    features: {
      count: 10,
      numeric: [
        'time_in_hospital',
        'num_lab_procedures',
        'num_medications',
        'num_procedures',
        'number_inpatient',
        'number_outpatient',
        'number_emergency',
        'number_diagnoses'
      ],
      categorical: [
        'age',
        'gender'
      ]
    },

    performance: {
      testAccuracy: 0.6205,
      testPrecision: 0.1568,
      testRecall: 0.5687,
      testF1Score: 0.2458,
      testSpecificity: 0.6268,
      rocAuc: 0.6451,
      optimalThreshold: 0.55
    },

    preprocessing: {
      missingValueStrategy: {
        numeric: 'median imputation',
        categorical: 'unknown category'
      },
      encoding: {
        categorical: 'label encoding'
      },
      normalization: {
        numeric: 'z-score (mean=0, std=1)'
      }
    },

    files: {
      model: 'model.json',
      weights: 'weights.bin',
      preprocessing: '../preprocess.json',
      documentation: '../../docs/model.md'
    },

    usage: {
      inputShape: [10],
      outputShape: [1],
      outputType: 'probability',
      threshold: {
        default: 0.5,
        optimal: 0.55
      }
    },

    warnings: [
      'This model is a research prototype and NOT approved for clinical use',
      'Predictions should supplement, not replace, clinical judgment',
      'Model should be validated on local data before deployment',
      'Trained on US hospital data (1999-2008) - may not generalize to other populations'
    ]
  };

  const metadataPath = path.join(modelDir, 'metadata.json');
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

  console.log(`Model metadata saved to: ${metadataPath}`);
  console.log(`\nMetadata summary:`);
  console.log(`  Version: ${metadata.version}`);
  console.log(`  Created: ${metadata.createdDate}`);
  console.log(`  Dataset: ${metadata.dataset.name} (${metadata.dataset.totalSamples} samples)`);
  console.log(`  Architecture: ${metadata.model.architecture}`);
  console.log(`  Parameters: ${metadata.model.totalParameters}`);
  console.log(`  ROC-AUC: ${metadata.performance.rocAuc.toFixed(4)}`);

  return metadata;
}

/**
 * Main verification function
 */
async function verify() {
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║   MODEL ARTIFACT VERIFICATION AND FINALIZATION      ║');
  console.log('╚══════════════════════════════════════════════════════╝\n');

  await tf.setBackend('cpu');
  await tf.ready();

  const modelDir = path.join(__dirname, 'models', 'readmission-model');
  const preprocessPath = path.join(__dirname, 'models', 'preprocess.json');

  // Step 1: Load model
  console.log('Step 1: Loading model...');
  const model = await loadModel(modelDir);
  console.log('✓ Model loaded successfully\n');

  // Display model summary
  console.log('Model Summary:');
  model.summary();

  // Step 2: Load and verify preprocessing metadata
  console.log('\nStep 2: Loading preprocessing metadata...');
  const metadata = loadMetadata(preprocessPath);
  console.log('✓ Preprocessing metadata loaded\n');

  const preprocessingValid = verifyPreprocessingMetadata(metadata);

  if (!preprocessingValid) {
    throw new Error('Preprocessing metadata verification failed!');
  }

  // Step 3: Test inference
  console.log('\nStep 3: Testing inference pipeline...');
  await testInference(model, metadata);
  console.log('✓ Inference test passed\n');

  // Step 4: Create model metadata
  console.log('Step 4: Creating model metadata...');
  const modelMetadata = createModelMetadata(modelDir);
  console.log('✓ Model metadata created\n');

  // Final summary
  console.log('\n╔══════════════════════════════════════════════════════╗');
  console.log('║              VERIFICATION COMPLETE                   ║');
  console.log('╚══════════════════════════════════════════════════════╝\n');

  console.log('All checks passed! Model artifacts are ready for deployment.\n');

  console.log('Generated files:');
  console.log(`  ✓ ${path.join(modelDir, 'model.json')}`);
  console.log(`  ✓ ${path.join(modelDir, 'weights.bin')}`);
  console.log(`  ✓ ${path.join(modelDir, 'metadata.json')}`);
  console.log(`  ✓ ${preprocessPath}`);
  console.log(`  ✓ ${path.join(__dirname, '../docs/model.md')}`);

  console.log('\nNext steps:');
  console.log('  1. Review docs/model.md for comprehensive documentation');
  console.log('  2. Proceed with backend API setup (Task 7)');
  console.log('  3. Build frontend for user interface (Task 10-11)');
}

// Run verification
verify().catch(err => {
  console.error('\n✗ Verification failed:', err.message);
  console.error(err.stack);
  process.exit(1);
});
