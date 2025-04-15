from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add these imports at the top
from fastapi.middleware.cors import CORSMiddleware
from collabCompass import *

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class BrandRequest(BaseModel):
    campaign_goals: str
    target_age: str
    content_preferences: str
    key_points: str

class InfluencerRequest(BaseModel):
    content_style: str
    follower_count: int
    topics: str

class PitchRequest(BaseModel):
    brand: BrandRequest
    style: str = "Casual with emojis"

# Initialize components at startup
@app.on_event("startup")
async def startup_event():
    global analyzer, manager
    analyzer = UnifiedAnalyzer()
    manager = CollaborationManager()
    
    # Load mock data
    influencers = pd.DataFrame([{
        'content_style': 'sustainable beauty',
        'follower_count': 15000,
        'topics': 'eco-friendly, skincare'
    }])
    
    brands = pd.DataFrame([{
        'campaign_goals': 'Sustainable beauty launch',
        'target_age': '18-35',
        'content_preferences': 'tutorials, reviews'
    }])
    
    analyzer.process_profiles(influencers, brands)
    analyzer.enable_rag([{
        'video_description': 'Check out our sustainable products 🌱',
        'like_count': 1500
    }])

# API Endpoints
@app.post("/match")
async def find_matches(brand: BrandRequest, query: str):
    try:
        brand_df = pd.DataFrame([brand.dict()])
        analyzer.process_profiles(pd.DataFrame(), brand_df)
        matches = analyzer.find_matches(0, query)
        return {"matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-pitch")
async def generate_pitch(request: PitchRequest):
    try:
        brand_info = {
            'name': 'GreenBeauty Co',
            'campaign': request.brand.campaign_goals,
            'key_points': request.brand.key_points,
            'relevant_content_inspiration': "Sustainable beauty products"
        }
        pitch = manager.generate_pitch(brand_info, request.style)
        return {"pitch": pitch}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
