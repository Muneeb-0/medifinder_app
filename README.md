# MediFinder AI Backend API

FastAPI backend service for AI-powered medicine intelligence using OpenAI.

## Features

- 🔍 Medicine composition analysis
- 💊 Medical use information
- 📋 Dosage guidelines
- 🔄 Alternative medicine suggestions
- 🛡️ Professional medical prompts
- ⚡ Fast and efficient API

## Setup Instructions

### 1. Install Python Dependencies

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

1. Copy the example environment file:
```bash
copy .env.example .env
# On macOS/Linux:
cp .env.example .env
```

2. Open `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**To get your OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and paste it in `.env` file

### 3. Run the Server

```bash
# Make sure you're in the backend directory and virtual environment is activated
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base URL**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/`

## API Endpoints

### Health Check
```
GET /
```
Returns server status.

### Medicine Intelligence
```
POST /api/medicine-intelligence
```

**Request Body:**
```json
{
  "medicine_name": "Paracetamol"
}
```

**Response:**
```json
{
  "medicine_name": "Paracetamol",
  "composition": {
    "active_ingredients": ["Paracetamol: 500mg"],
    "inactive_ingredients": ["Starch", "Cellulose"]
  },
  "medical_use": "Pain relief and fever reduction",
  "dosage": {
    "adult_dosage": "500-1000mg",
    "pediatric_dosage": "10-15mg/kg",
    "frequency": "Every 4-6 hours",
    "duration": "As needed"
  },
  "alternatives": [
    {
      "name": "Acetaminophen",
      "composition": "Paracetamol 500mg",
      "similarity_reason": "Same active ingredient",
      "manufacturer": "Various"
    }
  ],
  "disclaimer": "This information is for educational purposes only..."
}
```

## Testing the API

### Using curl:
```bash
curl -X POST "http://localhost:8000/api/medicine-intelligence" \
  -H "Content-Type: application/json" \
  -d '{"medicine_name": "Paracetamol"}'
```

### Using Python:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/medicine-intelligence",
    json={"medicine_name": "Paracetamol"}
)
print(response.json())
```

### Using the Interactive Docs:
Visit `http://localhost:8000/docs` in your browser for interactive API documentation.

## Troubleshooting

### OpenAI API Key Error
- Make sure `.env` file exists in the `backend` directory
- Verify the API key is correct and has credits
- Check for extra spaces or quotes in the `.env` file

### Port Already in Use
- Change the port in `main.py` or use:
```bash
uvicorn main:app --port 8001
```

### CORS Issues
- The backend is configured to allow all origins for development
- For production, update CORS settings in `main.py`

## Production Deployment

For production deployment:
1. Set `allow_origins` in CORS middleware to your Flutter app's domain
2. Use environment variables for sensitive data
3. Enable HTTPS
4. Use a production ASGI server like Gunicorn with Uvicorn workers
5. Set up proper logging and monitoring

## License

Part of the MediFinder App project.

