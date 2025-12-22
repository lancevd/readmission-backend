import path from 'path';
import { fileURLToPath } from 'url';
import {
  loadCSV,
  preprocessData,
  extractFeaturesAndLabels,
  saveMetadata,
  FEATURE_CONFIG
} from './preprocess.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('=== PREPROCESSING PIPELINE TEST ===\n');

// Load dataset
const csvPath = path.join(__dirname, '../data/raw/diabetic_data.csv');
console.log('Loading dataset...');
const rawData = loadCSV(csvPath);
console.log(`Loaded ${rawData.length} records\n`);

// Run preprocessing
console.log('Running preprocessing pipeline...');
const { data: processedData, metadata } = preprocessData(rawData);
console.log('Preprocessing complete!\n');

// Extract features and labels
const { features, labels } = extractFeaturesAndLabels(processedData);
console.log(`Extracted ${features.length} samples with ${features[0].length} features\n`);

// Display metadata
console.log('=== PREPROCESSING METADATA ===\n');

console.log('Features:');
console.log('  Numeric:', metadata.features.numeric.join(', '));
console.log('  Categorical:', metadata.features.categorical.join(', '));
console.log();

console.log('Medians (for missing value imputation):');
Object.entries(metadata.medians).forEach(([feature, median]) => {
  console.log(`  ${feature}: ${median.toFixed(2)}`);
});
console.log();

console.log('Normalization parameters (mean, std):');
Object.entries(metadata.means).forEach(([feature, mean]) => {
  const std = metadata.stds[feature];
  console.log(`  ${feature}: mean=${mean.toFixed(2)}, std=${std.toFixed(2)}`);
});
console.log();

console.log('Categorical encodings:');
Object.entries(metadata.encodings).forEach(([feature, encoding]) => {
  console.log(`  ${feature}:`, encoding);
});
console.log();

// Label distribution
const positiveSamples = labels.filter(label => label === 1).length;
const negativeSamples = labels.filter(label => label === 0).length;
console.log('=== LABEL DISTRIBUTION ===');
console.log(`Positive (readmitted <30 days): ${positiveSamples} (${((positiveSamples / labels.length) * 100).toFixed(2)}%)`);
console.log(`Negative (not readmitted <30 days): ${negativeSamples} (${((negativeSamples / labels.length) * 100).toFixed(2)}%)`);
console.log();

// Sample preprocessed data
console.log('=== SAMPLE PREPROCESSED DATA (first 3 records) ===');
const allFeatures = [...FEATURE_CONFIG.numeric, ...FEATURE_CONFIG.categorical];
for (let i = 0; i < Math.min(3, features.length); i++) {
  console.log(`\nRecord ${i + 1}:`);
  console.log(`  Label: ${labels[i]}`);
  console.log('  Features:');
  features[i].forEach((val, idx) => {
    console.log(`    ${allFeatures[idx]}: ${val.toFixed(4)}`);
  });
}
console.log();

// Save metadata
const metadataPath = path.join(__dirname, 'models', 'preprocess.json');
saveMetadata(metadata, metadataPath);

console.log('\n=== PREPROCESSING TEST COMPLETE ===');
