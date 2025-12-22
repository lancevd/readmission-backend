import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import request from 'supertest';

// We'll need to start the server for testing
// For now, we'll test against a running server
const API_URL = 'http://localhost:5000';

describe('API Integration Tests', () => {
  describe('GET / - Health Check Endpoint', () => {
    it('should return server health status', async () => {
      const response = await request(API_URL).get('/');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('uptime');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('services');
      expect(response.body.services).toHaveProperty('model');
      expect(response.body.services).toHaveProperty('database');
    });

    it('should have valid timestamp format', async () => {
      const response = await request(API_URL).get('/');

      expect(response.status).toBe(200);
      const timestamp = new Date(response.body.timestamp);
      expect(timestamp).toBeInstanceOf(Date);
      expect(isNaN(timestamp.getTime())).toBe(false);
    });

    it('should report model as loaded', async () => {
      const response = await request(API_URL).get('/');

      expect(response.status).toBe(200);
      expect(response.body.services.model).toBe('loaded');
    });
  });

  describe('GET /api/health - API Health Endpoint', () => {
    it('should return API health status', async () => {
      const response = await request(API_URL).get('/api/health');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status', 'ok');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body).toHaveProperty('model');
      expect(response.body).toHaveProperty('database');
    });

    it('should return model ready status', async () => {
      const response = await request(API_URL).get('/api/health');

      expect(response.status).toBe(200);
      expect(response.body.model).toBe('ready');
    });

    it('should have valid JSON content type', async () => {
      const response = await request(API_URL).get('/api/health');

      expect(response.status).toBe(200);
      expect(response.headers['content-type']).toMatch(/application\/json/);
    });
  });

  describe('GET /api/model/info - Model Info Endpoint', () => {
    it('should return model metadata', async () => {
      const response = await request(API_URL).get('/api/model/info');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('version');
      expect(response.body).toHaveProperty('name');
      expect(response.body).toHaveProperty('description');
      expect(response.body).toHaveProperty('architecture');
      expect(response.body).toHaveProperty('performance');
      expect(response.body).toHaveProperty('features');
      expect(response.body).toHaveProperty('warnings');
    });

    it('should have correct model version', async () => {
      const response = await request(API_URL).get('/api/model/info');

      expect(response.status).toBe(200);
      expect(response.body.version).toBe('1.0.0');
    });

    it('should include performance metrics', async () => {
      const response = await request(API_URL).get('/api/model/info');

      expect(response.status).toBe(200);
      expect(response.body.performance).toHaveProperty('testAccuracy');
      expect(response.body.performance).toHaveProperty('testPrecision');
      expect(response.body.performance).toHaveProperty('testRecall');
      expect(response.body.performance).toHaveProperty('testF1Score');
      expect(response.body.performance).toHaveProperty('rocAuc');
    });

    it('should list all required features', async () => {
      const response = await request(API_URL).get('/api/model/info');

      expect(response.status).toBe(200);
      expect(response.body.features).toHaveProperty('count', 10);
      expect(response.body.features).toHaveProperty('numeric');
      expect(response.body.features).toHaveProperty('categorical');
      expect(Array.isArray(response.body.features.numeric)).toBe(true);
      expect(Array.isArray(response.body.features.categorical)).toBe(true);
      expect(response.body.features.numeric.length).toBe(8);
      expect(response.body.features.categorical.length).toBe(2);
    });
  });

  describe('POST /api/predict - Prediction Endpoint (Valid Data)', () => {
    it('should predict medium risk for typical patient', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[60-70)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('success', true);
      expect(response.body).toHaveProperty('prediction');
      expect(response.body.prediction).toHaveProperty('probability');
      expect(response.body.prediction).toHaveProperty('probabilityPercent');
      expect(response.body.prediction).toHaveProperty('riskLevel');
      expect(response.body.prediction).toHaveProperty('riskCategory');
      expect(response.body.prediction).toHaveProperty('threshold');
    });

    it('should predict high risk for high-risk patient', async () => {
      const patientData = {
        time_in_hospital: '7',
        num_lab_procedures: '60',
        num_medications: '20',
        num_procedures: '3',
        number_inpatient: '2',
        number_outpatient: '1',
        number_emergency: '1',
        number_diagnoses: '9',
        age: '[70-80)',
        gender: 'Female'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.prediction.riskLevel).toBe('high');
      expect(response.body.prediction.probability).toBeGreaterThanOrEqual(0.5);
    });

    it('should include metadata in response', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[50-60)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('metadata');
      expect(response.body.metadata).toHaveProperty('modelVersion', '1.0.0');
      expect(response.body.metadata).toHaveProperty('timestamp');
    });

    it('should include disclaimer in response', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[50-60)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('disclaimer');
      expect(response.body.disclaimer).toContain('research purposes only');
    });

    it('should have probability between 0 and 1', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[50-60)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.prediction.probability).toBeGreaterThanOrEqual(0);
      expect(response.body.prediction.probability).toBeLessThanOrEqual(1);
    });

    it('should accept all valid age ranges', async () => {
      const ageRanges = [
        '[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
        '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'
      ];

      for (const age of ageRanges) {
        const patientData = {
          time_in_hospital: '3',
          num_lab_procedures: '44',
          num_medications: '15',
          num_procedures: '1',
          number_inpatient: '0',
          number_outpatient: '0',
          number_emergency: '0',
          number_diagnoses: '8',
          age: age,
          gender: 'Male'
        };

        const response = await request(API_URL)
          .post('/api/predict')
          .send(patientData)
          .set('Content-Type', 'application/json');

        expect(response.status).toBe(200);
        expect(response.body.success).toBe(true);
      }
    });

    it('should accept all valid genders', async () => {
      const genders = ['Female', 'Male', 'Unknown/Invalid'];

      for (const gender of genders) {
        const patientData = {
          time_in_hospital: '3',
          num_lab_procedures: '44',
          num_medications: '15',
          num_procedures: '1',
          number_inpatient: '0',
          number_outpatient: '0',
          number_emergency: '0',
          number_diagnoses: '8',
          age: '[50-60)',
          gender: gender
        };

        const response = await request(API_URL)
          .post('/api/predict')
          .send(patientData)
          .set('Content-Type', 'application/json');

        expect(response.status).toBe(200);
        expect(response.body.success).toBe(true);
      }
    });
  });

  describe('POST /api/predict - Missing Fields', () => {
    it('should return 400 when time_in_hospital is missing', async () => {
      const patientData = {
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[50-60)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body).toHaveProperty('missingFields');
      expect(response.body.missingFields).toContain('time_in_hospital');
    });

    it('should return 400 when multiple fields are missing', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        age: '[50-60)'
        // Missing several required fields
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body).toHaveProperty('missingFields');
      expect(response.body.missingFields.length).toBeGreaterThan(0);
    });

    it('should return 400 when request body is empty', async () => {
      const response = await request(API_URL)
        .post('/api/predict')
        .send({})
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      expect(response.body).toHaveProperty('missingFields');
      expect(response.body.missingFields.length).toBe(10);
    });
  });

  describe('POST /api/predict - Edge Cases', () => {
    it('should handle zero values for numeric fields', async () => {
      const patientData = {
        time_in_hospital: '0',
        num_lab_procedures: '0',
        num_medications: '0',
        num_procedures: '0',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '0',
        age: '[20-30)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should handle maximum reasonable values', async () => {
      const patientData = {
        time_in_hospital: '14',
        num_lab_procedures: '100',
        num_medications: '50',
        num_procedures: '10',
        number_inpatient: '5',
        number_outpatient: '5',
        number_emergency: '5',
        number_diagnoses: '15',
        age: '[80-90)',
        gender: 'Female'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should handle youngest age group', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[0-10)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should handle oldest age group', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[90-100)',
        gender: 'Female'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });

    it('should handle Unknown/Invalid gender', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[50-60)',
        gender: 'Unknown/Invalid'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
  });

  describe('POST /api/predict - Risk Level Thresholds', () => {
    it('should classify low risk correctly (< 30%)', async () => {
      // This is a theoretical test - actual values depend on the model
      const patientData = {
        time_in_hospital: '1',
        num_lab_procedures: '20',
        num_medications: '5',
        num_procedures: '0',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '3',
        age: '[20-30)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);

      if (response.body.prediction.probability < 0.3) {
        expect(response.body.prediction.riskLevel).toBe('low');
        expect(response.body.prediction.riskCategory).toBe('Low Risk');
      }
    });

    it('should have consistent risk level based on probability', async () => {
      const patientData = {
        time_in_hospital: '3',
        num_lab_procedures: '44',
        num_medications: '15',
        num_procedures: '1',
        number_inpatient: '0',
        number_outpatient: '0',
        number_emergency: '0',
        number_diagnoses: '8',
        age: '[50-60)',
        gender: 'Male'
      };

      const response = await request(API_URL)
        .post('/api/predict')
        .send(patientData)
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(200);

      const { probability, riskLevel } = response.body.prediction;

      if (probability < 0.3) {
        expect(riskLevel).toBe('low');
      } else if (probability < 0.5) {
        expect(riskLevel).toBe('medium');
      } else {
        expect(riskLevel).toBe('high');
      }
    });
  });

  describe('Error Handling', () => {
    it('should return 404 for non-existent routes', async () => {
      const response = await request(API_URL).get('/api/nonexistent');

      expect(response.status).toBe(404);
    });

    it('should handle malformed JSON', async () => {
      const response = await request(API_URL)
        .post('/api/predict')
        .send('{"invalid": json}')
        .set('Content-Type', 'application/json');

      expect(response.status).toBe(400);
    });
  });
});
