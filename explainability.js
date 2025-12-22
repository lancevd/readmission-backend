import * as tf from '@tensorflow/tfjs';

/**
 * Calculate permutation feature importance for a specific prediction
 * This measures how much each feature contributes to the prediction
 */
export async function calculateFeatureImportance(model, patientData, preprocessMetadata, preprocessFunc) {
  const { preprocessData, extractFeaturesAndLabels } = preprocessFunc;

  // Get baseline prediction
  const baselineArray = [{
    ...patientData,
    readmitted: 'NO' // Placeholder
  }];

  const { data: baselineProcessed } = preprocessData(baselineArray, preprocessMetadata);
  const { features: baselineFeatures } = extractFeaturesAndLabels(baselineProcessed);
  const baselineTensor = tf.tensor2d(baselineFeatures);
  const baselinePrediction = model.predict(baselineTensor).dataSync()[0];
  baselineTensor.dispose();

  // Feature names in order
  const featureNames = [
    'time_in_hospital',
    'num_lab_procedures',
    'num_medications',
    'num_procedures',
    'number_inpatient',
    'number_outpatient',
    'number_emergency',
    'number_diagnoses',
    'age',
    'gender'
  ];

  const importanceScores = [];

  // For each feature, calculate its importance
  for (let i = 0; i < featureNames.length; i++) {
    const featureName = featureNames[i];

    // Create modified data with this feature set to median/mode
    const modifiedData = { ...patientData };

    // Set to median for numeric features, or most common value for categorical
    if (preprocessMetadata.features.numeric.includes(featureName)) {
      modifiedData[featureName] = String(preprocessMetadata.medians[featureName] || 0);
    } else {
      // For categorical features, use a neutral value
      if (featureName === 'age') {
        modifiedData[featureName] = '[50-60)'; // Middle age range
      } else if (featureName === 'gender') {
        modifiedData[featureName] = 'Unknown/Invalid';
      }
    }

    // Get prediction with modified feature
    const modifiedArray = [{
      ...modifiedData,
      readmitted: 'NO'
    }];

    const { data: modifiedProcessed } = preprocessData(modifiedArray, preprocessMetadata);
    const { features: modifiedFeatures } = extractFeaturesAndLabels(modifiedProcessed);
    const modifiedTensor = tf.tensor2d(modifiedFeatures);
    const modifiedPrediction = model.predict(modifiedTensor).dataSync()[0];
    modifiedTensor.dispose();

    // Importance is the absolute change in prediction
    const importance = Math.abs(baselinePrediction - modifiedPrediction);

    importanceScores.push({
      feature: featureName,
      importance: importance,
      contribution: (baselinePrediction - modifiedPrediction) * 100 // Percentage point contribution
    });
  }

  // Sort by importance (descending)
  importanceScores.sort((a, b) => b.importance - a.importance);

  return {
    scores: importanceScores,
    baselineProbability: baselinePrediction
  };
}

/**
 * Generate human-readable explanation for a prediction
 */
export function generateExplanation(patientData, prediction, featureImportance) {
  const { probability, riskLevel } = prediction;
  const topFeatures = featureImportance.scores.slice(0, 3); // Top 3 features

  // Get actual values for top features
  const featureDescriptions = {
    time_in_hospital: (val) => `${val} day${val !== '1' ? 's' : ''} in hospital`,
    num_lab_procedures: (val) => `${val} lab procedure${val !== '1' ? 's' : ''}`,
    num_medications: (val) => `${val} medication${val !== '1' ? 's' : ''}`,
    num_procedures: (val) => `${val} procedure${val !== '1' ? 's' : ''}`,
    number_inpatient: (val) => `${val} prior inpatient visit${val !== '1' ? 's' : ''}`,
    number_outpatient: (val) => `${val} prior outpatient visit${val !== '1' ? 's' : ''}`,
    number_emergency: (val) => `${val} prior emergency visit${val !== '1' ? 's' : ''}`,
    number_diagnoses: (val) => `${val} diagnos${val !== '1' ? 'es' : 'is'}`,
    age: (val) => `age range ${val}`,
    gender: (val) => `gender: ${val}`
  };

  // Build explanation
  let explanation = `This patient has a ${riskLevel} risk of readmission (${(probability * 100).toFixed(1)}% probability). `;

  // Add top contributing factors
  if (topFeatures.length > 0) {
    explanation += `The main factors contributing to this assessment are: `;

    const factorTexts = topFeatures.map((feat, idx) => {
      const value = patientData[feat.feature];
      const description = featureDescriptions[feat.feature]
        ? featureDescriptions[feat.feature](value)
        : `${feat.feature}: ${value}`;

      const contributionText = feat.contribution > 0
        ? `increasing risk by ${Math.abs(feat.contribution).toFixed(1)}%`
        : `reducing risk by ${Math.abs(feat.contribution).toFixed(1)}%`;

      return `${description} (${contributionText})`;
    });

    if (factorTexts.length === 1) {
      explanation += factorTexts[0];
    } else if (factorTexts.length === 2) {
      explanation += `${factorTexts[0]} and ${factorTexts[1]}`;
    } else {
      explanation += factorTexts.slice(0, -1).join(', ') + ', and ' + factorTexts[factorTexts.length - 1];
    }

    explanation += '.';
  }

  // Add clinical recommendations based on risk level
  if (riskLevel === 'high') {
    explanation += ' Consider intensive discharge planning, patient education, medication reconciliation, and scheduling follow-up appointments within 7 days.';
  } else if (riskLevel === 'medium') {
    explanation += ' Consider enhanced discharge planning and follow-up care to reduce readmission risk.';
  } else {
    explanation += ' Standard discharge protocols are appropriate, but ensure patient understanding of discharge instructions.';
  }

  return explanation;
}

/**
 * Get friendly feature names for display
 */
export function getFeatureDisplayName(featureName) {
  const displayNames = {
    time_in_hospital: 'Time in Hospital',
    num_lab_procedures: 'Lab Procedures',
    num_medications: 'Medications',
    num_procedures: 'Procedures',
    number_inpatient: 'Prior Inpatient Visits',
    number_outpatient: 'Prior Outpatient Visits',
    number_emergency: 'Prior Emergency Visits',
    number_diagnoses: 'Number of Diagnoses',
    age: 'Age Range',
    gender: 'Gender'
  };

  return displayNames[featureName] || featureName;
}
