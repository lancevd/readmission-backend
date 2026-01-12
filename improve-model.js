/**
 * Model Improvement Strategies
 *
 * This file contains code snippets and approaches to improve model fairness
 * and performance based on the fairness analysis findings.
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Strategy 1: Age-Specific Thresholds
 * Apply different decision thresholds for different age groups
 */
export function getAgeSpecificThreshold(age) {
  const thresholds = {
    0: 0.30,  // [0-10)   - Lower threshold (reduce false positives)
    1: 0.35,  // [10-20)
    2: 0.40,  // [20-30)  - Best performing group, lower threshold acceptable
    3: 0.45,  // [30-40)
    4: 0.50,  // [40-50)  - Standard threshold
    5: 0.50,  // [50-60)
    6: 0.55,  // [60-70)
    7: 0.60,  // [70-80)  - Higher threshold (reduce false positives)
    8: 0.65,  // [80-90)
    9: 0.70   // [90-100) - Highest threshold for oldest patients
  };

  return thresholds[age] || 0.5;
}

/**
 * Strategy 2: Confidence-Based Predictions
 * Only make high-confidence predictions, defer uncertain cases to human review
 */
export function getConfidenceBasedPrediction(probability, age) {
  const threshold = getAgeSpecificThreshold(age);
  const confidenceMargin = 0.15;

  if (probability < threshold - confidenceMargin) {
    return {
      riskLevel: 'low',
      confidence: 'high',
      requiresReview: false
    };
  } else if (probability > threshold + confidenceMargin) {
    return {
      riskLevel: 'high',
      confidence: 'high',
      requiresReview: false
    };
  } else {
    return {
      riskLevel: 'uncertain',
      confidence: 'low',
      requiresReview: true  // Flag for human review
    };
  }
}

/**
 * Strategy 3: Weighted Ensemble
 * Combine predictions from multiple models or approaches
 */
export function ensemblePrediction(mlProbability, ruleBasedScore, clinicalRiskFactors) {
  // Weight ML more for middle-aged patients, rules more for elderly
  const age = clinicalRiskFactors.age;
  const mlWeight = age < 6 ? 0.7 : 0.4;  // Reduce ML weight for elderly
  const ruleWeight = 1 - mlWeight;

  const ensembleProbability = (mlProbability * mlWeight) + (ruleBasedScore * ruleWeight);

  return {
    probability: ensembleProbability,
    mlComponent: mlProbability,
    ruleComponent: ruleBasedScore,
    weights: { ml: mlWeight, rule: ruleWeight }
  };
}

/**
 * Strategy 4: Rule-Based Risk Adjustments
 * Apply clinical knowledge to adjust ML predictions
 */
export function applyRuleBasedAdjustments(probability, patientData) {
  let adjustedProb = probability;

  // High-risk indicators
  if (patientData.number_inpatient >= 2) {
    adjustedProb = Math.min(1.0, adjustedProb + 0.10);  // Increase risk
  }

  if (patientData.time_in_hospital >= 7) {
    adjustedProb = Math.min(1.0, adjustedProb + 0.08);
  }

  if (patientData.number_emergency >= 2) {
    adjustedProb = Math.min(1.0, adjustedProb + 0.12);
  }

  // Low-risk indicators
  if (patientData.number_inpatient === 0 &&
      patientData.number_outpatient === 0 &&
      patientData.number_emergency === 0) {
    adjustedProb = Math.max(0.0, adjustedProb - 0.10);  // Decrease risk
  }

  return {
    originalProbability: probability,
    adjustedProbability: adjustedProb,
    adjustment: adjustedProb - probability
  };
}

/**
 * Strategy 5: Calibration per Demographic Group
 * Recalibrate probabilities to match true rates per subgroup
 */
export class DemographicCalibrator {
  constructor() {
    // These would be learned from validation data
    this.calibrationMaps = {
      // Age group calibration (maps raw probability to calibrated probability)
      age: {
        0: { slope: 0.8, intercept: 0.1 },
        1: { slope: 0.85, intercept: 0.08 },
        2: { slope: 1.2, intercept: -0.05 },  // Best group, boost confidence
        3: { slope: 1.1, intercept: 0.0 },
        4: { slope: 1.0, intercept: 0.0 },
        5: { slope: 0.95, intercept: 0.02 },
        6: { slope: 0.9, intercept: 0.05 },
        7: { slope: 0.85, intercept: 0.08 },
        8: { slope: 0.75, intercept: 0.12 },   // Reduce over-prediction
        9: { slope: 0.65, intercept: 0.15 }    // Strongest reduction for oldest
      }
    };
  }

  calibrate(probability, age) {
    const params = this.calibrationMaps.age[age] || { slope: 1.0, intercept: 0.0 };
    const calibrated = (probability * params.slope) + params.intercept;

    // Ensure bounded [0, 1]
    return Math.max(0, Math.min(1, calibrated));
  }
}

/**
 * Strategy 6: Uncertainty Quantification
 * Estimate prediction uncertainty using dropout at inference time
 */
export async function predictWithUncertainty(model, inputTensor, numSamples = 10) {
  const predictions = [];

  // Make multiple predictions with dropout enabled
  for (let i = 0; i < numSamples; i++) {
    const pred = model.predict(inputTensor, { training: true });  // Keep dropout on
    predictions.push(pred.dataSync()[0]);
    pred.dispose();
  }

  // Calculate mean and standard deviation
  const mean = predictions.reduce((a, b) => a + b) / predictions.length;
  const variance = predictions.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / predictions.length;
  const std = Math.sqrt(variance);

  return {
    probability: mean,
    uncertainty: std,
    confidence: 1 - std,  // Lower std = higher confidence
    predictions: predictions
  };
}

/**
 * Strategy 7: Feature Importance Thresholding
 * Only trust predictions where important features are informative
 */
export function getFeatureBasedConfidence(patientData) {
  // Key features identified from explainability analysis
  const importantFeatures = [
    'number_inpatient',
    'time_in_hospital',
    'number_emergency'
  ];

  // Check if patient has informative values for important features
  let informativeCount = 0;
  if (patientData.number_inpatient > 0) informativeCount++;
  if (patientData.time_in_hospital > 3) informativeCount++;
  if (patientData.number_emergency > 0) informativeCount++;

  const confidence = informativeCount / importantFeatures.length;

  return {
    confidence,
    informativeFeatures: informativeCount,
    totalImportantFeatures: importantFeatures.length,
    recommendation: confidence < 0.3 ? 'defer_to_human' : 'use_prediction'
  };
}

/**
 * COMBINED IMPROVEMENT STRATEGY
 * Use all strategies together for best results
 */
export async function improvedPrediction(model, patientData, preprocessMetadata) {
  // 1. Get base ML prediction
  const { preprocessData, extractFeaturesAndLabels } = await import('./preprocess.js');
  const patientArray = [{ ...patientData, readmitted: 'NO' }];
  const { data: processedData } = preprocessData(patientArray, preprocessMetadata);
  const { features } = extractFeaturesAndLabels(processedData);

  const inputTensor = tf.tensor2d(features);
  const predictions = model.predict(inputTensor);
  const rawProbability = predictions.dataSync()[0];
  inputTensor.dispose();
  predictions.dispose();

  // 2. Apply demographic calibration
  const calibrator = new DemographicCalibrator();
  const calibratedProb = calibrator.calibrate(rawProbability, processedData[0].age);

  // 3. Apply rule-based adjustments
  const { adjustedProbability } = applyRuleBasedAdjustments(calibratedProb, patientData);

  // 4. Get age-specific threshold
  const threshold = getAgeSpecificThreshold(processedData[0].age);

  // 5. Determine risk level with confidence
  const prediction = getConfidenceBasedPrediction(adjustedProbability, processedData[0].age);

  // 6. Get feature-based confidence
  const featureConfidence = getFeatureBasedConfidence(patientData);

  return {
    // Probabilities
    rawProbability,
    calibratedProbability: calibratedProb,
    finalProbability: adjustedProbability,

    // Risk assessment
    riskLevel: prediction.riskLevel,
    threshold,

    // Confidence metrics
    modelConfidence: prediction.confidence,
    featureConfidence: featureConfidence.confidence,
    requiresReview: prediction.requiresReview || featureConfidence.recommendation === 'defer_to_human',

    // Metadata
    age: processedData[0].age,
    adjustments: {
      calibration: calibratedProb - rawProbability,
      rules: adjustedProbability - calibratedProb
    }
  };
}

export default {
  getAgeSpecificThreshold,
  getConfidenceBasedPrediction,
  ensemblePrediction,
  applyRuleBasedAdjustments,
  DemographicCalibrator,
  predictWithUncertainty,
  getFeatureBasedConfidence,
  improvedPrediction
};
