import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import mongoose from 'mongoose';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-cpu';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { loadMetadata } from './preprocess.js';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Global model and metadata storage
let model = null;
let preprocessMetadata = null;
let modelMetadata = null;

/**
 * Load model from filesystem
 */
async function loadModel(modelDir) {
  const modelJsonPath = path.join(modelDir, 'model.json');
  const weightsPath = path.join(modelDir, 'weights.bin');

  console.log('Loading model...');
  console.log(`  Model: ${modelJsonPath}`);
  console.log(`  Weights: ${weightsPath}`);

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
  const loadedModel = await tf.models.modelFromJSON(modelData.modelTopology);

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

  loadedModel.setWeights(weightValues);

  console.log('✓ Model loaded successfully');
  return loadedModel;
}

/**
 * Initialize MongoDB connection
 */
async function connectDatabase() {
  try {
    const mongoUri = process.env.MONGODB_URI || 'mongodb://localhost:27017/readmission';

    console.log('Connecting to MongoDB...');
    console.log(`  URI: ${mongoUri.replace(/\/\/.*@/, '//<credentials>@')}`);

    await mongoose.connect(mongoUri);

    console.log('✓ MongoDB connected successfully');
  } catch (error) {
    console.error('✗ MongoDB connection failed:', error.message);
    console.log('  Server will continue without database connection');
    // Don't throw - allow server to run without DB for now
  }
}

/**
 * Initialize ML model and metadata
 */
async function initializeModel() {
  try {
    // Set TensorFlow.js backend
    await tf.setBackend('cpu');
    await tf.ready();
    console.log(`TensorFlow.js backend: ${tf.getBackend()}`);

    // Load model
    const modelDir = process.env.MODEL_PATH || path.join(__dirname, 'models', 'readmission-model');
    model = await loadModel(modelDir);

    // Load preprocessing metadata
    const preprocessPath = process.env.PREPROCESS_PATH || path.join(__dirname, 'models', 'preprocess.json');
    console.log(`Loading preprocessing metadata from: ${preprocessPath}`);
    preprocessMetadata = loadMetadata(preprocessPath);
    console.log('✓ Preprocessing metadata loaded');

    // Load model metadata
    const metadataPath = path.join(modelDir, 'metadata.json');
    if (fs.existsSync(metadataPath)) {
      modelMetadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
      console.log(`✓ Model metadata loaded (version ${modelMetadata.version})`);
    }

    console.log('✓ Model initialization complete\n');
  } catch (error) {
    console.error('✗ Failed to initialize model:', error);
    throw error;
  }
}

/**
 * Create and configure Express app
 */
function createApp() {
  const app = express();

  // Middleware
  app.use(cors({
    origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
    credentials: true
  }));

  app.use(express.json());
  app.use(express.urlencoded({ extended: true }));

  // Request logging middleware
  app.use((req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] ${req.method} ${req.path}`);
    next();
  });

  // Health check endpoint
  app.get('/', (req, res) => {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      services: {
        api: 'operational',
        model: model ? 'loaded' : 'not loaded',
        database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
      }
    };

    if (modelMetadata) {
      health.model = {
        version: modelMetadata.version,
        name: modelMetadata.name,
        rocAuc: modelMetadata.performance.rocAuc
      };
    }

    res.json(health);
  });

  // Model info endpoint
  app.get('/api/model/info', (req, res) => {
    if (!modelMetadata) {
      return res.status(503).json({
        error: 'Model metadata not available'
      });
    }

    res.json({
      version: modelMetadata.version,
      name: modelMetadata.name,
      description: modelMetadata.description,
      createdDate: modelMetadata.createdDate,
      architecture: modelMetadata.model.architecture,
      parameters: modelMetadata.model.totalParameters,
      performance: modelMetadata.performance,
      features: modelMetadata.features,
      warnings: modelMetadata.warnings
    });
  });

  // API status endpoint
  app.get('/api/health', (req, res) => {
    res.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      model: model ? 'ready' : 'not loaded',
      database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
    });
  });

  // Prediction endpoint
  app.post('/api/predict', async (req, res, next) => {
    try {
      // Check if model is loaded
      if (!model || !preprocessMetadata) {
        return res.status(503).json({
          error: 'Model not available',
          message: 'The prediction model is not loaded. Please try again later.'
        });
      }

      // Validate request body
      const patientData = req.body;

      // Required fields
      const requiredFields = [
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

      // Check for missing fields
      const missingFields = requiredFields.filter(field => !(field in patientData));
      if (missingFields.length > 0) {
        return res.status(400).json({
          error: 'Missing required fields',
          missingFields
        });
      }

      // Import preprocessing on demand
      const { preprocessData, extractFeaturesAndLabels } = await import('./preprocess.js');

      // Prepare patient data for preprocessing
      const patientArray = [{
        ...patientData,
        readmitted: 'NO' // Placeholder, not used in prediction
      }];

      // Preprocess data
      const { data: processedData } = preprocessData(patientArray, preprocessMetadata);
      const { features } = extractFeaturesAndLabels(processedData);

      // Convert to tensor
      const inputTensor = tf.tensor2d(features);

      // Make prediction
      const predictions = model.predict(inputTensor);
      const probability = predictions.dataSync()[0];

      // Determine risk level
      let riskLevel;
      let riskCategory;
      if (probability >= 0.5) {
        riskLevel = 'high';
        riskCategory = 'High Risk';
      } else if (probability >= 0.3) {
        riskLevel = 'medium';
        riskCategory = 'Medium Risk';
      } else {
        riskLevel = 'low';
        riskCategory = 'Low Risk';
      }

      // Clean up tensors
      inputTensor.dispose();
      predictions.dispose();

      // Log prediction (without patient data for privacy)
      console.log(`[PREDICTION] Risk: ${riskLevel}, Probability: ${(probability * 100).toFixed(2)}%`);

      // Return prediction
      res.json({
        success: true,
        prediction: {
          probability: parseFloat(probability.toFixed(4)),
          probabilityPercent: parseFloat((probability * 100).toFixed(2)),
          riskLevel,
          riskCategory,
          threshold: 0.5
        },
        metadata: {
          modelVersion: modelMetadata?.version || 'unknown',
          timestamp: new Date().toISOString()
        },
        disclaimer: 'This prediction is for research purposes only and should not be used for clinical decision-making.'
      });

    } catch (error) {
      console.error('[PREDICTION ERROR]', error.message);
      next(error);
    }
  });

  // 404 handler
  app.use((req, res) => {
    res.status(404).json({
      error: 'Endpoint not found',
      path: req.path
    });
  });

  // Error handling middleware
  app.use((err, req, res, next) => {
    console.error('Error:', err);

    res.status(err.status || 500).json({
      error: err.message || 'Internal server error',
      ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
    });
  });

  return app;
}

/**
 * Start the server
 */
async function startServer() {
  console.log('╔══════════════════════════════════════════════════════╗');
  console.log('║     READMISSION PREDICTION API - STARTING UP        ║');
  console.log('╚══════════════════════════════════════════════════════╝\n');

  try {
    // Initialize database
    await connectDatabase();

    // Initialize model
    await initializeModel();

    // Create Express app
    const app = createApp();

    // Start server
    const PORT = process.env.PORT || 5000;
    app.listen(PORT, () => {
      console.log('╔══════════════════════════════════════════════════════╗');
      console.log('║            SERVER STARTED SUCCESSFULLY              ║');
      console.log('╚══════════════════════════════════════════════════════╝\n');
      console.log(`🚀 Server running on port ${PORT}`);
      console.log(`   Environment: ${process.env.NODE_ENV || 'development'}`);
      console.log(`   Health check: http://localhost:${PORT}/`);
      console.log(`   API health: http://localhost:${PORT}/api/health`);
      console.log(`   Model info: http://localhost:${PORT}/api/model/info`);
      console.log('\n   Press Ctrl+C to stop\n');
    });
  } catch (error) {
    console.error('✗ Failed to start server:', error);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n\nShutting down gracefully...');

  if (mongoose.connection.readyState === 1) {
    await mongoose.connection.close();
    console.log('✓ MongoDB connection closed');
  }

  console.log('✓ Server stopped');
  process.exit(0);
});

// Start the server
startServer();

// Export for testing
export { createApp, model, preprocessMetadata, modelMetadata };
