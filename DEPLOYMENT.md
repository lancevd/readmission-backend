# Backend Deployment Guide

## Overview
This is the backend API for the Patient Readmission Risk Prediction application. It includes:
- Express.js REST API
- TensorFlow.js ML model (~8KB)
- MongoDB Atlas integration (optional)

## Prerequisites
- Node.js 18+
- MongoDB Atlas account (optional, app works without it)

## Local Testing

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Update values as needed

3. **Start the server:**
   ```bash
   npm start
   ```

4. **Test endpoints:**
   - Health: http://localhost:5000/
   - API Health: http://localhost:5000/api/health
   - Model Info: http://localhost:5000/api/model/info

## Deployment Options

### Option 1: Render (Recommended)

1. **Create account:** https://render.com
2. **Create new Web Service**
3. **Connect your Git repository**
4. **Configure:**
   - **Name:** readmission-backend
   - **Environment:** Node
   - **Build Command:** `npm install`
   - **Start Command:** `npm start`
   - **Instance Type:** Free

5. **Environment Variables (in Render dashboard):**
   ```
   PORT=5000
   NODE_ENV=production
   MONGODB_URI=<your-mongodb-atlas-connection-string>
   CORS_ORIGIN=https://your-frontend-domain.vercel.app
   MODEL_PATH=./models/readmission-model
   PREPROCESS_PATH=./models/preprocess.json
   ```

6. **Deploy:** Click "Create Web Service"

7. **Get your backend URL:** `https://readmission-backend-xxxx.onrender.com`

### Option 2: Railway

1. **Create account:** https://railway.app
2. **New Project → Deploy from GitHub**
3. **Configure:**
   - **Start Command:** `npm start`
   - **Environment Variables:** (same as above)

### Option 3: Leapcell

1. **Create account:** https://leapcell.io
2. **Create new project**
3. **Deploy from Git**
4. **Set environment variables** (same as above)

## Important Notes

### CORS Configuration
After deploying both frontend and backend, update:
```
CORS_ORIGIN=https://your-actual-frontend-url.vercel.app
```

### MongoDB (Optional)
- If you don't set `MONGODB_URI`, the server will run without database
- Current connection string in `.env` is from original project (update with your own)

### Model Files
- The `models/` folder MUST be included in deployment
- Total size: ~8KB (very small, no issues)
- TensorFlow.js will use CPU inference (no GPU needed)

### Free Tier Limitations
- **Render:** May sleep after 15 min inactivity (takes 30s to wake up)
- **Railway:** 500 hours/month free
- **Leapcell:** Check their free tier limits

## Endpoints

### GET /
Health check with full system status

### GET /api/health
Quick API status

### GET /api/model/info
Model metadata and performance metrics

### POST /api/predict
Make a prediction

**Request Body:**
```json
{
  "time_in_hospital": "3",
  "num_lab_procedures": "50",
  "num_medications": "15",
  "num_procedures": "2",
  "number_inpatient": "0",
  "number_outpatient": "2",
  "number_emergency": "1",
  "number_diagnoses": "8",
  "age": "[60-70)",
  "gender": "Male"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "probability": 0.4422,
    "probabilityPercent": 44.22,
    "riskLevel": "medium",
    "riskCategory": "Medium Risk",
    "threshold": 0.5
  },
  "metadata": {
    "modelVersion": "1.0.0",
    "timestamp": "2025-12-22T..."
  },
  "disclaimer": "..."
}
```

## Troubleshooting

### Issue: "Model not found"
- Ensure `models/` folder is included in deployment
- Check `MODEL_PATH` environment variable

### Issue: "CORS error"
- Update `CORS_ORIGIN` to match your frontend URL
- Include `https://` in the URL

### Issue: Server not starting
- Check logs in your deployment platform
- Verify all dependencies installed correctly
- Ensure `package.json` scripts are correct

## Project Structure

```
readmission-backend/
├── server.js              # Main Express server
├── preprocess.js          # Data preprocessing
├── models/                # ML model files (CRITICAL!)
│   ├── preprocess.json
│   └── readmission-model/
│       ├── model.json
│       ├── weights.bin
│       └── metadata.json
├── package.json
├── .env                   # Local environment variables (DO NOT COMMIT)
└── .env.example           # Template for deployment
```

## Performance
- Cold start: ~2-5 seconds (on free tier)
- Warm request: ~100-300ms
- Prediction time: ~50-100ms
- Memory usage: ~100-150MB
