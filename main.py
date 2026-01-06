from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file in the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Fallback: If load_dotenv didn't work, try reading .env file directly
if not os.getenv("OPENAI_API_KEY") and env_path.exists():
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() == 'OPENAI_API_KEY':
                        os.environ['OPENAI_API_KEY'] = value.strip()
                        break
    except Exception as e:
        print(f"⚠️ Error reading .env file: {e}")

# Debug: Check if API key is loaded (remove in production)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    # Strip whitespace for display
    api_key_clean = api_key.strip()
    print(f"✅ OpenAI API key loaded successfully (length: {len(api_key_clean)})")
    if api_key != api_key_clean:
        print(f"⚠️ WARNING: API key had trailing whitespace, it will be stripped automatically")
else:
    print("⚠️ WARNING: OpenAI API key not found in environment variables")
    print(f"   Looking for .env file at: {env_path.absolute()}")
    print(f"   .env file exists: {env_path.exists()}")

# Initialize FastAPI app
app = FastAPI(
    title="MediFinder AI API",
    description="AI-powered medicine intelligence API for finding medicine alternatives",
    version="1.0.0"
)

# CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client - will be initialized when needed
def get_openai_client():
    """Get OpenAI client, initializing it if needed"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    # Strip whitespace (newlines, spaces) from API key to prevent header errors
    api_key = api_key.strip()
    return OpenAI(api_key=api_key)


# Request/Response Models
class MedicineSearchRequest(BaseModel):
    medicine_name: str


class MedicineComposition(BaseModel):
    active_ingredients: List[str]
    inactive_ingredients: Optional[List[str]] = None


class DosageInfo(BaseModel):
    adult_dosage: str
    pediatric_dosage: Optional[str] = None
    frequency: str
    duration: Optional[str] = None


class AlternativeMedicine(BaseModel):
    name: str
    composition: str
    similarity_reason: str
    manufacturer: Optional[str] = None


class MedicineIntelligenceResponse(BaseModel):
    medicine_name: str
    composition: MedicineComposition
    medical_use: str
    dosage: DosageInfo
    alternatives: List[AlternativeMedicine]
    disclaimer: str


# Medical prompt template
MEDICAL_PROMPT_TEMPLATE = """You are a highly qualified medical information specialist with expertise in pharmacology and pharmaceutical composition. Your task is to provide accurate, concise, and professional information about medicines.

For the medicine: "{medicine_name}"

Provide a detailed analysis in the following structured format:

1. COMPOSITION:
   - List all active ingredients with their exact dosages
   - List inactive ingredients (excipients) if relevant
   - Format: "Active Ingredient: Dosage" (e.g., "Paracetamol: 500mg")

2. MEDICAL USE:
   - Primary therapeutic indications
   - Conditions it treats
   - Mechanism of action (brief)
   - Format: Short, clear sentences (2-3 sentences max)

3. DOSAGE INFORMATION:
   - Adult dosage: Standard adult dose and frequency
   - Pediatric dosage: If applicable (age-specific if needed)
   - Frequency: How often to take (e.g., "Every 6-8 hours")
   - Duration: Typical treatment duration if applicable
   - Format: Clear, concise instructions

4. ALTERNATIVES:
   Find medicines with the SAME active ingredients and similar composition that work for the same medical conditions but have DIFFERENT brand names.
   - List 3-5 alternatives
   - For each alternative, provide:
     * Brand name
     * Composition (active ingredients)
     * Why it's similar (same active ingredients, same use case)
     * Manufacturer (if known)
   - Format: Short, clear entries

IMPORTANT GUIDELINES:
- Be highly accurate and fact-based
- Use medical terminology correctly
- Keep responses concise and clear
- Only suggest alternatives with identical active ingredients
- If medicine information is not available or uncertain, state that clearly
- Do not provide medical advice beyond factual information
- Format the response in a structured, easy-to-parse manner

Response Format (JSON-like structure):
{{
  "composition": {{
    "active_ingredients": ["Ingredient1: Dosage", "Ingredient2: Dosage"],
    "inactive_ingredients": ["Ingredient1", "Ingredient2"]
  }},
  "medical_use": "Brief description of medical use and indications",
  "dosage": {{
    "adult_dosage": "Standard adult dose",
    "pediatric_dosage": "Age-specific dose if applicable",
    "frequency": "How often to take",
    "duration": "Treatment duration"
  }},
  "alternatives": [
    {{
      "name": "Alternative Brand Name",
      "composition": "Active ingredients",
      "similarity_reason": "Why it's similar",
      "manufacturer": "Manufacturer name if known"
    }}
  ]
}}
"""


def parse_openai_response(response_text: str) -> dict:
    """Parse OpenAI response and extract structured data"""
    import json
    import re
    
    # Try to extract JSON from the response
    # Look for JSON-like structure in the response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    
    if json_match:
        try:
            # Clean and parse JSON
            json_str = json_match.group(0)
            # Remove markdown code blocks if present
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Fallback: Try to extract information manually
    result = {
        "composition": {"active_ingredients": [], "inactive_ingredients": []},
        "medical_use": "",
        "dosage": {
            "adult_dosage": "",
            "pediatric_dosage": "",
            "frequency": "",
            "duration": ""
        },
        "alternatives": []
    }
    
    # Extract composition
    comp_match = re.search(r'COMPOSITION[:\s]*(.*?)(?=MEDICAL USE|$)', response_text, re.IGNORECASE | re.DOTALL)
    if comp_match:
        comp_text = comp_match.group(1)
        # Extract active ingredients
        active_matches = re.findall(r'([A-Za-z\s]+):\s*(\d+\s*(?:mg|g|ml|%))', comp_text, re.IGNORECASE)
        result["composition"]["active_ingredients"] = [f"{ing.strip()}: {dose}" for ing, dose in active_matches]
    
    # Extract medical use
    use_match = re.search(r'MEDICAL USE[:\s]*(.*?)(?=DOSAGE|ALTERNATIVES|$)', response_text, re.IGNORECASE | re.DOTALL)
    if use_match:
        result["medical_use"] = use_match.group(1).strip()
    
    # Extract dosage
    dosage_match = re.search(r'DOSAGE[:\s]*(.*?)(?=ALTERNATIVES|$)', response_text, re.IGNORECASE | re.DOTALL)
    if dosage_match:
        dosage_text = dosage_match.group(1)
        adult_match = re.search(r'adult[:\s]*(.*?)(?=pediatric|frequency|$)', dosage_text, re.IGNORECASE)
        if adult_match:
            result["dosage"]["adult_dosage"] = adult_match.group(1).strip()
        freq_match = re.search(r'frequency[:\s]*(.*?)(?=duration|$)', dosage_text, re.IGNORECASE)
        if freq_match:
            result["dosage"]["frequency"] = freq_match.group(1).strip()
    
    # Extract alternatives
    alt_match = re.search(r'ALTERNATIVES[:\s]*(.*?)$', response_text, re.IGNORECASE | re.DOTALL)
    if alt_match:
        alt_text = alt_match.group(1)
        # Try to extract alternative entries
        alt_entries = re.findall(r'([A-Za-z\s]+)[:\s]*Composition[:\s]*(.*?)(?=[A-Z]|$)', alt_text, re.IGNORECASE | re.DOTALL)
        for name, comp in alt_entries[:5]:  # Limit to 5
            result["alternatives"].append({
                "name": name.strip(),
                "composition": comp.strip()[:100],  # Limit length
                "similarity_reason": "Same active ingredients",
                "manufacturer": ""
            })
    
    return result


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MediFinder AI API",
        "version": "1.0.0"
    }


@app.post("/api/medicine-intelligence", response_model=MedicineIntelligenceResponse)
async def get_medicine_intelligence(request: MedicineSearchRequest):
    """
    Get comprehensive medicine intelligence including composition, medical use, dosage, and alternatives
    """
    if not request.medicine_name or not request.medicine_name.strip():
        raise HTTPException(status_code=400, detail="Medicine name is required")
    
    # Check if OpenAI API key is configured
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY in .env file"
        )
    
    try:
        # Initialize OpenAI client
        openai_client = get_openai_client()
        
        # Create the prompt
        prompt = MEDICAL_PROMPT_TEMPLATE.format(medicine_name=request.medicine_name.strip())
        
        print(f"🔍 Requesting medicine intelligence for: {request.medicine_name.strip()}")
        
        # Call OpenAI API
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better accuracy
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical information specialist. Provide accurate, concise, and structured information about medicines. Always format your response as valid JSON when possible."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=2000,
                response_format={"type": "json_object"}  # Request JSON response
            )
            print(f"✅ OpenAI API call successful")
        except Exception as openai_error:
            print(f"❌ OpenAI API error: {type(openai_error).__name__}: {str(openai_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error: {str(openai_error)}"
            )
        
        # Extract response
        response_text = response.choices[0].message.content
        
        # Parse the response
        try:
            import json
            parsed_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback parsing if JSON parsing fails
            parsed_data = parse_openai_response(response_text)
        
        # Build response
        composition = MedicineComposition(
            active_ingredients=parsed_data.get("composition", {}).get("active_ingredients", []),
            inactive_ingredients=parsed_data.get("composition", {}).get("inactive_ingredients", [])
        )
        
        dosage = DosageInfo(
            adult_dosage=parsed_data.get("dosage", {}).get("adult_dosage", "Consult healthcare provider"),
            pediatric_dosage=parsed_data.get("dosage", {}).get("pediatric_dosage"),
            frequency=parsed_data.get("dosage", {}).get("frequency", "As directed"),
            duration=parsed_data.get("dosage", {}).get("duration")
        )
        
        alternatives = [
            AlternativeMedicine(
                name=alt.get("name", ""),
                composition=alt.get("composition", ""),
                similarity_reason=alt.get("similarity_reason", "Same active ingredients"),
                manufacturer=alt.get("manufacturer")
            )
            for alt in parsed_data.get("alternatives", [])
        ]
        
        return MedicineIntelligenceResponse(
            medicine_name=request.medicine_name.strip(),
            composition=composition,
            medical_use=parsed_data.get("medical_use", "Information not available"),
            dosage=dosage,
            alternatives=alternatives,
            disclaimer="This information is for educational purposes only. Always consult with a qualified healthcare professional before taking any medication. Dosage and usage may vary based on individual health conditions."
        )
        
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in medicine intelligence endpoint:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Full traceback:\n{error_trace}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing medicine intelligence: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    import os
    # Use PORT environment variable for deployment (Railway/Render), fallback to 8000 for local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

