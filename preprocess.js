import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';

/**
 * Feature configuration based on EDA
 */
const FEATURE_CONFIG = {
  // Numeric features
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
  // Categorical features
  categorical: [
    'age',
    'gender'
  ],
  // Target variable
  target: 'readmitted'
};

/**
 * Load and parse CSV file
 */
function loadCSV(filePath) {
  const csvContent = fs.readFileSync(filePath, 'utf8');
  const parsed = Papa.parse(csvContent, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false
  });
  return parsed.data;
}

/**
 * Convert target variable to binary
 * <30 -> 1 (positive: readmitted within 30 days)
 * NO or >30 -> 0 (negative: not readmitted within 30 days)
 */
function createBinaryTarget(data) {
  return data.map(row => ({
    ...row,
    target: row[FEATURE_CONFIG.target] === '<30' ? 1 : 0
  }));
}

/**
 * Calculate median for numeric array
 */
function calculateMedian(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/**
 * Calculate mean
 */
function calculateMean(values) {
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate standard deviation
 */
function calculateStd(values, mean) {
  const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Handle missing values in numeric features
 * Missing values are represented as '?' or empty strings
 */
function handleMissingNumeric(data, feature, metadata) {
  // Extract valid numeric values
  const validValues = data
    .map(row => row[feature])
    .filter(val => val !== '?' && val !== '' && val !== null && val !== undefined)
    .map(val => parseFloat(val))
    .filter(val => !isNaN(val));

  // Calculate median
  const median = calculateMedian(validValues);
  metadata.medians[feature] = median;

  // Replace missing values with median
  return data.map(row => {
    const val = row[feature];
    if (val === '?' || val === '' || val === null || val === undefined) {
      return { ...row, [feature]: median };
    }
    const parsed = parseFloat(val);
    return { ...row, [feature]: isNaN(parsed) ? median : parsed };
  });
}

/**
 * Handle missing values in categorical features
 */
function handleMissingCategorical(data, feature) {
  return data.map(row => {
    const val = row[feature];
    if (val === '?' || val === '' || val === null || val === undefined) {
      return { ...row, [feature]: 'unknown' };
    }
    return row;
  });
}

/**
 * Encode categorical features using label encoding
 */
function encodeCategorical(data, feature, metadata) {
  // Get unique values
  const uniqueValues = [...new Set(data.map(row => row[feature]))].sort();

  // Create encoding map
  const encoding = {};
  uniqueValues.forEach((val, idx) => {
    encoding[val] = idx;
  });

  metadata.encodings[feature] = encoding;

  // Apply encoding
  return data.map(row => ({
    ...row,
    [feature]: encoding[row[feature]]
  }));
}

/**
 * Normalize numeric features using z-score normalization (mean=0, std=1)
 */
function normalizeNumeric(data, feature, metadata) {
  // Extract values
  const values = data.map(row => parseFloat(row[feature]));

  // Calculate mean and std
  const mean = calculateMean(values);
  const std = calculateStd(values, mean);

  metadata.means[feature] = mean;
  metadata.stds[feature] = std;

  // Normalize: (x - mean) / std
  // Handle case where std is 0 (all values are the same)
  return data.map(row => ({
    ...row,
    [feature]: std === 0 ? 0 : (parseFloat(row[feature]) - mean) / std
  }));
}

/**
 * Main preprocessing pipeline
 * Returns preprocessed data and metadata
 */
function preprocessData(data, metadata = null) {
  const isTraining = metadata === null;

  if (isTraining) {
    // Initialize metadata for training
    metadata = {
      features: {
        numeric: FEATURE_CONFIG.numeric,
        categorical: FEATURE_CONFIG.categorical
      },
      medians: {},
      means: {},
      stds: {},
      encodings: {}
    };
  }

  let processedData = [...data];

  // Create binary target
  processedData = createBinaryTarget(processedData);

  // Handle missing values for numeric features
  for (const feature of FEATURE_CONFIG.numeric) {
    if (isTraining) {
      processedData = handleMissingNumeric(processedData, feature, metadata);
    } else {
      // Use saved median for inference
      processedData = processedData.map(row => {
        const val = row[feature];
        if (val === '?' || val === '' || val === null || val === undefined) {
          return { ...row, [feature]: metadata.medians[feature] };
        }
        const parsed = parseFloat(val);
        return { ...row, [feature]: isNaN(parsed) ? metadata.medians[feature] : parsed };
      });
    }
  }

  // Handle missing values for categorical features
  for (const feature of FEATURE_CONFIG.categorical) {
    processedData = handleMissingCategorical(processedData, feature);
  }

  // Encode categorical features
  for (const feature of FEATURE_CONFIG.categorical) {
    if (isTraining) {
      processedData = encodeCategorical(processedData, feature, metadata);
    } else {
      // Use saved encoding for inference
      processedData = processedData.map(row => ({
        ...row,
        [feature]: metadata.encodings[feature][row[feature]] ?? 0
      }));
    }
  }

  // Normalize numeric features
  for (const feature of FEATURE_CONFIG.numeric) {
    if (isTraining) {
      processedData = normalizeNumeric(processedData, feature, metadata);
    } else {
      // Use saved mean and std for inference
      const mean = metadata.means[feature];
      const std = metadata.stds[feature];
      processedData = processedData.map(row => ({
        ...row,
        [feature]: std === 0 ? 0 : (parseFloat(row[feature]) - mean) / std
      }));
    }
  }

  return { data: processedData, metadata };
}

/**
 * Extract features and labels as arrays for model training
 */
function extractFeaturesAndLabels(processedData) {
  const allFeatures = [...FEATURE_CONFIG.numeric, ...FEATURE_CONFIG.categorical];

  const features = processedData.map(row =>
    allFeatures.map(feature => parseFloat(row[feature]))
  );

  const labels = processedData.map(row => row.target);

  return { features, labels };
}

/**
 * Save preprocessing metadata to JSON file
 */
function saveMetadata(metadata, outputPath) {
  const dir = path.dirname(outputPath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  fs.writeFileSync(outputPath, JSON.stringify(metadata, null, 2));
  console.log(`Preprocessing metadata saved to ${outputPath}`);
}

/**
 * Load preprocessing metadata from JSON file
 */
function loadMetadata(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(content);
}

export {
  FEATURE_CONFIG,
  loadCSV,
  preprocessData,
  extractFeaturesAndLabels,
  saveMetadata,
  loadMetadata
};
