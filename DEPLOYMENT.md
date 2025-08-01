# DocAssist AI Deployment Guide

## Backend Deployment (Render)

### 1. Deploy Backend to Render

1. **Connect your GitHub repository to Render**
2. **Create a new Web Service**
3. **Configure the service:**
   - **Name**: `docassist-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Environment Variables**:
     - `FLASK_ENV`: `production`
     - `PYTHON_VERSION`: `3.8.0`

4. **Deploy the service**
5. **Note the URL**: `https://your-app-name.onrender.com`

### 2. Update Frontend API URLs

Update the API URLs in your frontend files to point to your Render backend:

**In `frontend/public/index.html`:**
```javascript
const response = await fetch('https://your-app-name.onrender.com/predict', {
```

**In `frontend/public/app.html`:**
```javascript
const response = await fetch('https://your-app-name.onrender.com/predict', {
```

## Frontend Deployment (Vercel)

### 1. Deploy Frontend to Vercel

1. **Connect your GitHub repository to Vercel**
2. **Configure the deployment:**
   - **Framework Preset**: `Other`
   - **Root Directory**: `./` (root of repository)
   - **Build Command**: Leave empty (not needed for static files)
   - **Output Directory**: `frontend/public`

3. **Environment Variables**:
   - `API_URL`: `https://your-app-name.onrender.com`

4. **Deploy**

### 2. Vercel Configuration

The `vercel.json` file is already configured to serve static files from `frontend/public/`.

## Troubleshooting

### Common Issues:

1. **404 Error on Vercel**:
   - Make sure your repository structure is correct
   - Ensure `frontend/public/` contains your HTML files
   - Check that `vercel.json` is in the root directory

2. **CORS Errors**:
   - Backend should have CORS configured (already done)
   - Frontend should use the correct backend URL

3. **API Connection Issues**:
   - Verify the backend URL is correct
   - Check that the backend is running on Render
   - Ensure environment variables are set correctly

### Testing Deployment:

1. **Test Backend**: Visit `https://your-app-name.onrender.com/health`
2. **Test Frontend**: Visit your Vercel URL
3. **Test API**: Try submitting a prediction through the frontend

## File Structure for Deployment:

```
DOCASSIST-AI/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── final_model_pipeline.pkl
├── frontend/
│   └── public/
│       ├── index.html
│       └── app.html
├── vercel.json
└── render.yaml
```

## Environment Variables:

### Render (Backend):
- `FLASK_ENV`: `production`
- `PYTHON_VERSION`: `3.8.0`

### Vercel (Frontend):
- `API_URL`: `https://your-app-name.onrender.com` 