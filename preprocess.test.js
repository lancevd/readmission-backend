import { describe, it, expect, beforeAll } from 'vitest';
import path from 'path';
import { fileURLToPath } from 'url';
import {
  FEATURE_CONFIG,
  loadCSV,
  preprocessData,
  extractFeaturesAndLabels,
  saveMetadata,
  loadMetadata
} from './preprocess.js';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Preprocessing Pipeline Tests', () => {
  let rawData;
  let testMetadataPath;

  beforeAll(() => {
    // Load dataset once for all tests
    const csvPath = path.join(__dirname, '../data/raw/diabetic_data.csv');
    rawData = loadCSV(csvPath);
    testMetadataPath = path.join(__dirname, 'models', 'test-preprocess.json');
  });

  describe('CSV Loading', () => {
    it('should load CSV file successfully', () => {
      expect(rawData).toBeDefined();
      expect(Array.isArray(rawData)).toBe(true);
      expect(rawData.length).toBeGreaterThan(0);
    });

    it('should have all required columns', () => {
      const firstRow = rawData[0];
      expect(firstRow).toHaveProperty('readmitted');

      // Check all numeric features exist
      FEATURE_CONFIG.numeric.forEach(feature => {
        expect(firstRow).toHaveProperty(feature);
      });

      // Check all categorical features exist
      FEATURE_CONFIG.categorical.forEach(feature => {
        expect(firstRow).toHaveProperty(feature);
      });
    });

    it('should load the expected number of records', () => {
      expect(rawData.length).toBe(101766);
    });
  });

  describe('Missing Value Handling', () => {
    it('should handle missing numeric values with median', () => {
      // Create test data with missing values
      const testData = [
        { time_in_hospital: '5', num_lab_procedures: '40', num_medications: '15', num_procedures: '1',
          number_inpatient: '0', number_outpatient: '0', number_emergency: '0', number_diagnoses: '8',
          age: '[50-60)', gender: 'Male', readmitted: 'NO' },
        { time_in_hospital: '?', num_lab_procedures: '50', num_medications: '20', num_procedures: '2',
          number_inpatient: '1', number_outpatient: '0', number_emergency: '0', number_diagnoses: '9',
          age: '[60-70)', gender: 'Female', readmitted: '<30' },
        { time_in_hospital: '3', num_lab_procedures: '', num_medications: '10', num_procedures: '0',
          number_inpatient: '0', number_outpatient: '1', number_emergency: '0', number_diagnoses: '7',
          age: '[40-50)', gender: 'Male', readmitted: 'NO' }
      ];

      const { data: processed } = preprocessData(testData);

      // Check that missing values are replaced
      expect(processed[1].time_in_hospital).toBeDefined();
      expect(typeof processed[1].time_in_hospital).toBe('number');
      expect(processed[2].num_lab_procedures).toBeDefined();
      expect(typeof processed[2].num_lab_procedures).toBe('number');
    });

    it('should handle missing categorical values with "unknown"', () => {
      const testData = [
        { time_in_hospital: '5', num_lab_procedures: '40', num_medications: '15', num_procedures: '1',
          number_inpatient: '0', number_outpatient: '0', number_emergency: '0', number_diagnoses: '8',
          age: '?', gender: '', readmitted: 'NO' }
      ];

      const { data: processed, metadata } = preprocessData(testData);

      // Check that 'unknown' is in the encoding (missing values were replaced)
      expect(metadata.encodings.age).toHaveProperty('unknown');
      expect(metadata.encodings.gender).toHaveProperty('unknown');

      // Check that the values are encoded (should be the encoding for 'unknown')
      expect(typeof processed[0].age).toBe('number');
      expect(typeof processed[0].gender).toBe('number');
    });
  });

  describe('Categorical Encoding', () => {
    it('should encode categorical features consistently', { timeout: 60000 }, () => {
      const { data: processed, metadata } = preprocessData(rawData);

      // Check that encodings are created
      expect(metadata.encodings).toHaveProperty('age');
      expect(metadata.encodings).toHaveProperty('gender');

      // Check that all encoded values are numbers
      processed.forEach(row => {
        expect(typeof row.age).toBe('number');
        expect(typeof row.gender).toBe('number');
      });
    });

    it('should create consistent encoding maps', { timeout: 60000 }, () => {
      const { metadata } = preprocessData(rawData);

      // Age should have 10 categories (plus possibly 'unknown')
      const ageEncodingKeys = Object.keys(metadata.encodings.age);
      expect(ageEncodingKeys.length).toBeGreaterThanOrEqual(10);

      // Gender should have 3 categories (Female, Male, Unknown/Invalid, plus possibly 'unknown')
      const genderEncodingKeys = Object.keys(metadata.encodings.gender);
      expect(genderEncodingKeys.length).toBeGreaterThanOrEqual(3);

      // Check that encodings are sequential integers starting from 0
      const ageValues = Object.values(metadata.encodings.age);
      expect(Math.min(...ageValues)).toBe(0);
      expect(Math.max(...ageValues)).toBe(ageValues.length - 1);
    });

    it('should apply same encoding during inference', () => {
      // Train on subset
      const trainData = rawData.slice(0, 1000);
      const { metadata } = preprocessData(trainData);

      // Infer on new data
      const inferData = rawData.slice(1000, 1010);
      const { data: inferProcessed } = preprocessData(inferData, metadata);

      // Check that encoded values match the training encoding
      inferProcessed.forEach(row => {
        expect(typeof row.age).toBe('number');
        expect(typeof row.gender).toBe('number');
        expect(row.age).toBeGreaterThanOrEqual(0);
        expect(row.gender).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('Numeric Normalization', () => {
    it('should normalize numeric features to have mean ~0 and std ~1', { timeout: 60000 }, () => {
      const { data: processed, metadata } = preprocessData(rawData);

      // Extract normalized values for each feature
      FEATURE_CONFIG.numeric.forEach(feature => {
        const values = processed.map(row => row[feature]);

        // Calculate mean and std of normalized values
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const std = Math.sqrt(
          values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length
        );

        // Mean should be very close to 0 (within floating point precision)
        expect(Math.abs(mean)).toBeLessThan(1e-10);

        // Std should be very close to 1
        expect(Math.abs(std - 1)).toBeLessThan(1e-10);

        // Check metadata contains mean and std
        expect(metadata.means).toHaveProperty(feature);
        expect(metadata.stds).toHaveProperty(feature);
      });
    });

    it('should apply same normalization during inference', () => {
      // Train on subset
      const trainData = rawData.slice(0, 1000);
      const { metadata } = preprocessData(trainData);

      // Infer on new data
      const inferData = rawData.slice(1000, 1010);
      const { data: inferProcessed } = preprocessData(inferData, metadata);

      // Check that values are normalized using training statistics
      inferProcessed.forEach(row => {
        FEATURE_CONFIG.numeric.forEach(feature => {
          expect(typeof row[feature]).toBe('number');
          expect(isNaN(row[feature])).toBe(false);
        });
      });
    });

    it('should handle edge case where std is 0', () => {
      const testData = [
        { time_in_hospital: '5', num_lab_procedures: '40', num_medications: '15', num_procedures: '1',
          number_inpatient: '0', number_outpatient: '0', number_emergency: '0', number_diagnoses: '8',
          age: '[50-60)', gender: 'Male', readmitted: 'NO' },
        { time_in_hospital: '5', num_lab_procedures: '40', num_medications: '15', num_procedures: '1',
          number_inpatient: '0', number_outpatient: '0', number_emergency: '0', number_diagnoses: '8',
          age: '[60-70)', gender: 'Female', readmitted: '<30' }
      ];

      const { data: processed } = preprocessData(testData);

      // All values should be 0 when std is 0 (all same values)
      processed.forEach(row => {
        FEATURE_CONFIG.numeric.forEach(feature => {
          expect(row[feature]).toBe(0);
        });
      });
    });
  });

  describe('Binary Target Creation', () => {
    it('should convert readmitted to binary target correctly', { timeout: 60000 }, () => {
      const { data: processed } = preprocessData(rawData);

      // Count targets
      let positiveCount = 0;
      let negativeCount = 0;

      processed.forEach(row => {
        expect(row).toHaveProperty('target');
        expect([0, 1]).toContain(row.target);

        if (row.target === 1) positiveCount++;
        else negativeCount++;
      });

      // Verify counts match expected distribution (~11% positive)
      const positivePercentage = (positiveCount / processed.length) * 100;
      expect(positivePercentage).toBeGreaterThan(10);
      expect(positivePercentage).toBeLessThan(12);
    });

    it('should map <30 to 1 and others to 0', () => {
      const testData = [
        { time_in_hospital: '5', num_lab_procedures: '40', num_medications: '15', num_procedures: '1',
          number_inpatient: '0', number_outpatient: '0', number_emergency: '0', number_diagnoses: '8',
          age: '[50-60)', gender: 'Male', readmitted: '<30' },
        { time_in_hospital: '3', num_lab_procedures: '50', num_medications: '20', num_procedures: '2',
          number_inpatient: '1', number_outpatient: '0', number_emergency: '0', number_diagnoses: '9',
          age: '[60-70)', gender: 'Female', readmitted: 'NO' },
        { time_in_hospital: '4', num_lab_procedures: '45', num_medications: '18', num_procedures: '1',
          number_inpatient: '0', number_outpatient: '1', number_emergency: '0', number_diagnoses: '7',
          age: '[40-50)', gender: 'Male', readmitted: '>30' }
      ];

      const { data: processed } = preprocessData(testData);

      expect(processed[0].target).toBe(1); // <30 -> 1
      expect(processed[1].target).toBe(0); // NO -> 0
      expect(processed[2].target).toBe(0); // >30 -> 0
    });
  });

  describe('Feature Extraction', () => {
    it('should extract features and labels correctly', () => {
      const { data: processed } = preprocessData(rawData.slice(0, 100));
      const { features, labels } = extractFeaturesAndLabels(processed);

      // Check shapes
      expect(features.length).toBe(100);
      expect(labels.length).toBe(100);
      expect(features[0].length).toBe(10); // 8 numeric + 2 categorical

      // Check all values are numbers
      features.forEach(row => {
        row.forEach(val => {
          expect(typeof val).toBe('number');
          expect(isNaN(val)).toBe(false);
        });
      });

      labels.forEach(label => {
        expect([0, 1]).toContain(label);
      });
    });
  });

  describe('Metadata Persistence', () => {
    it('should save and load metadata correctly', () => {
      const { metadata } = preprocessData(rawData.slice(0, 100));

      // Save metadata
      saveMetadata(metadata, testMetadataPath);

      // Load metadata
      const loadedMetadata = loadMetadata(testMetadataPath);

      // Verify structure
      expect(loadedMetadata).toHaveProperty('features');
      expect(loadedMetadata).toHaveProperty('medians');
      expect(loadedMetadata).toHaveProperty('means');
      expect(loadedMetadata).toHaveProperty('stds');
      expect(loadedMetadata).toHaveProperty('encodings');

      // Verify content matches
      expect(loadedMetadata.features).toEqual(metadata.features);
      expect(Object.keys(loadedMetadata.medians)).toEqual(Object.keys(metadata.medians));
      expect(Object.keys(loadedMetadata.means)).toEqual(Object.keys(metadata.means));

      // Clean up test file
      if (fs.existsSync(testMetadataPath)) {
        fs.unlinkSync(testMetadataPath);
      }
    });
  });

  describe('Preprocessing Determinism', () => {
    it('should produce same results when run multiple times', () => {
      const testData = rawData.slice(0, 1000);

      // Run preprocessing twice
      const { data: processed1, metadata: metadata1 } = preprocessData(testData);
      const { data: processed2, metadata: metadata2 } = preprocessData(testData);

      // Check metadata is identical
      expect(metadata1.medians).toEqual(metadata2.medians);
      expect(metadata1.means).toEqual(metadata2.means);
      expect(metadata1.stds).toEqual(metadata2.stds);
      expect(metadata1.encodings).toEqual(metadata2.encodings);

      // Check processed data is identical
      expect(processed1.length).toBe(processed2.length);
      for (let i = 0; i < processed1.length; i++) {
        FEATURE_CONFIG.numeric.forEach(feature => {
          expect(processed1[i][feature]).toBeCloseTo(processed2[i][feature], 10);
        });
        FEATURE_CONFIG.categorical.forEach(feature => {
          expect(processed1[i][feature]).toBe(processed2[i][feature]);
        });
        expect(processed1[i].target).toBe(processed2[i].target);
      }
    });

    it('should produce consistent results for inference', () => {
      // Train once
      const trainData = rawData.slice(0, 1000);
      const { metadata } = preprocessData(trainData);

      // Infer multiple times on same data
      const inferData = rawData.slice(1000, 1100);
      const { data: infer1 } = preprocessData(inferData, metadata);
      const { data: infer2 } = preprocessData(inferData, metadata);

      // Check results are identical
      expect(infer1.length).toBe(infer2.length);
      for (let i = 0; i < infer1.length; i++) {
        FEATURE_CONFIG.numeric.forEach(feature => {
          expect(infer1[i][feature]).toBeCloseTo(infer2[i][feature], 10);
        });
        FEATURE_CONFIG.categorical.forEach(feature => {
          expect(infer1[i][feature]).toBe(infer2[i][feature]);
        });
      }
    });
  });
});
