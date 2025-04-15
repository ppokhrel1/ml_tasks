import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

import faiss
import requests

# Configuration
PLATFORM = "tiktok"  # Choose "instagram" or "tiktok"
USE_MOCK_DATA = True  # Set to True if no API access

# API Credentials (Set these in environment variables)
INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN", "your_instagram_token")
TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY", "your_tiktok_key")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET", "your_tiktok_secret")

class SocialMediaAPI(ABC):
    @abstractmethod
    def get_user_profile(self, identifier: str) -> Dict:
        pass
    
    @abstractmethod
    def get_user_content(self, identifier: str) -> List[Dict]:
        pass

class TikTokAPI(SocialMediaAPI):
    def __init__(self):
        self.access_token = self._get_access_token()
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    def _get_access_token(self):
        auth_url = "https://open.tiktokapis.com/v2/oauth/token/"
        data = {
            "client_key": TIKTOK_CLIENT_KEY,
            "client_secret": TIKTOK_CLIENT_SECRET,
            "grant_type": "client_credentials"
        }
        response = requests.post(auth_url, data=data)
        return response.json().get('access_token')
    
    def get_user_profile(self, username: str) -> Dict:
        url = "https://open.tiktokapis.com/v2/user/info/"
        params = {"fields": "open_id,display_name,follower_count"}
        response = requests.get(url, headers=self.headers, params=params)
        return response.json().get('data', {})
    
    def get_user_content(self, open_id: str) -> List[Dict]:
        url = "https://open.tiktokapis.com/v2/video/list/"
        data = {
            "open_id": open_id,
            "max_count": 10,
            "fields": "id,video_description,like_count"
        }
        response = requests.post(url, headers=self.headers, json=data)
        return response.json().get('data', {}).get('videos', [])

class InstagramAPI(SocialMediaAPI):
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.instagram.com/v18.0"
    
    def get_user_profile(self, user_id: str) -> Dict:
        url = f"{self.base_url}/{user_id}"
        params = {"fields": "id,username,followers_count", "access_token": self.access_token}
        response = requests.get(url, params=params)
        return response.json()
    
    def get_user_content(self, user_id: str) -> List[Dict]:
        url = f"{self.base_url}/{user_id}/media"
        params = {
            "fields": "id,caption,like_count",
            "access_token": self.access_token,
            "limit": 10
        }
        response = requests.get(url, params=params)
        return response.json().get('data', [])

class MockSocialAPI(SocialMediaAPI):
    def get_user_profile(self, identifier: str) -> Dict:
        if PLATFORM == "tiktok":
            return {
                'open_id': 'mock_123',
                'display_name': 'EcoCreator',
                'follower_count': 15000
            }
        return {
            'id': '17841405793187218',
            'username': 'eco_demo',
            'followers_count': 12000
        }
    
    def get_user_content(self, identifier: str) -> List[Dict]:
        content_key = 'video_description' if PLATFORM == 'tiktok' else 'caption'
        return [{
            content_key: 'Check out our sustainable products 🌱',
            'like_count': 1500
        }]

class PlatformRAG:
    def __init__(self, model: SentenceTransformer):
        self.model = model
        self.index = faiss.IndexFlatIP(model.get_sentence_embedding_dimension())  # Changed to Inner Product
        self.content_key = 'video_description' if PLATFORM == 'tiktok' else 'caption'
        self.content_store = []
    
    def add_content(self, content: List[Dict]):
        texts = [p.get(self.content_key, '') for p in content]
        embeddings = self.model.encode(texts).astype('float32')  # Add float32 conversion
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.content_store.extend(content)
    
    def retrieve_relevant(self, query: str, k: int = 3) -> List[Dict]:
        query_embed = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embed)
        distances, indices = self.index.search(query_embed, k)
        return [self.content_store[i] for i in indices[0]]

        
class UnifiedAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rag = None
        self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())  

    def process_profiles(self, influencers: pd.DataFrame, brands: pd.DataFrame):
        influencer_texts = self._generate_influencer_texts(influencers)
        brand_texts = self._generate_brand_texts(brands)
        
        self.influencer_embeds = self.model.encode(influencer_texts)
        faiss.normalize_L2(self.influencer_embeds)
        self.index.add(self.influencer_embeds)
        
        self.brand_embeds = self.model.encode(brand_texts)
        faiss.normalize_L2(self.brand_embeds)
    
    def enable_rag(self, content: List[Dict]):
        self.rag = PlatformRAG(self.model)
        self.rag.add_content(content)
    
    def find_matches(self, brand_idx: int, query: str = None, top_k: int = 5) -> List:
        base_embed = self.brand_embeds[brand_idx]
        
        if self.rag and query:
            relevant = self.rag.retrieve_relevant(query)
            relevant_embeds = self.model.encode([p[self.rag.content_key] for p in relevant])
            blended_embed = np.mean([base_embed] + relevant_embeds.tolist(), axis=0)
        else:
            blended_embed = base_embed
        
        blended_embed = np.expand_dims(blended_embed, axis=0).astype('float32')
        faiss.normalize_L2(blended_embed)
        
        distances, indices = self.index.search(blended_embed, top_k)
        return [(idx, 1 - distance) for idx, distance in zip(indices[0], distances[0])]
    
    def _generate_influencer_texts(self, df: pd.DataFrame) -> List[str]:
        if PLATFORM == "tiktok":
            return [
                f"{row['content_style']} followers: {row['follower_count']} topics: {row['topics']}"
                for _, row in df.iterrows()
            ]
        return [
            f"{row['content_themes']} audience: {row['age_range']} {row['gender_distribution']}"
            for _, row in df.iterrows()
        ]
    
    def _generate_brand_texts(self, df: pd.DataFrame) -> List[str]:
        return [
            f"{row['campaign_goals']} target: {row['target_age']} {row['content_preferences']}"
            for _, row in df.iterrows()
        ]


class CollaborationManager:
    def __init__(self):
        import torch  # Add this import
        
        # Configure device
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialize model with proper device mapping
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Configure tokenizer properly
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for dialogue models
        
        # Configure pipeline with explicit settings
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            framework="pt"
        )
    
    def generate_pitch(self, brand_info: Dict, style: str = None) -> str:
        # Clean prompt structure
        prompt = f"""Generate a TikTok collaboration pitch for {brand_info['name']}:
Campaign: {brand_info['campaign']}
Key Features: {brand_info.get('key_points', '')}
Style: {style or 'Fun and engaging with emojis'}
Content Ideas:"""
        
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=500,  # Use tokens instead of length
                temperature=0.9,
                top_p=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            
            if not outputs:
                return "Error: No text generated"
                
            return outputs[0]['generated_text'].strip()
            
        except Exception as e:
            return f"Generation failed: {str(e)}"


# Execution Flow
if __name__ == "__main__":
    # Initialize API
    api = None
    try:
        if not USE_MOCK_DATA:
            if PLATFORM == "tiktok":
                api = TikTokAPI()
                identifier = "ecobeautycreator"  # TikTok username
            else:
                api = InstagramAPI(INSTAGRAM_ACCESS_TOKEN)
                identifier = "17841405793187218"  # Instagram user ID
    except Exception as e:
        print(f"API Error: {e}, using mock data")
        USE_MOCK_DATA = True
    
    if USE_MOCK_DATA:
        api = MockSocialAPI()
        identifier = "mock_user"

    # Fetch data
    profile = api.get_user_profile(identifier)
    content = api.get_user_content(identifier)

    # Prepare dataframes
    if PLATFORM == "tiktok":
        influencers = pd.DataFrame([{
            'id': profile.get('open_id', 'mock_123'),
            'content_style': 'sustainable beauty',
            'follower_count': profile.get('follower_count', 15000),
            'topics': 'eco-friendly, skincare'
        }])
    else:
        influencers = pd.DataFrame([{
            'id': profile.get('id', 'mock_123'),
            'content_themes': 'eco beauty',
            'age_range': '25-34',
            'gender_distribution': '90% female'
        }])

    brands = pd.DataFrame([{
        'campaign_goals': 'Sustainable beauty launch',
        'target_age': '18-35',
        'content_preferences': 'tutorials, reviews'
    }])

    # Process data
    analyzer = UnifiedAnalyzer()
    analyzer.process_profiles(influencers, brands)
    analyzer.enable_rag(content)

    # Find matches
    matches = analyzer.find_matches(
        brand_idx=0,
        query="Eco-friendly makeup products",
        top_k=3
    )
    print(f"Top Matches: {matches}")

    # Generate pitch
    manager = CollaborationManager()
    pitch = manager.generate_pitch({
        'name': 'GreenBeauty Co',
        'campaign': 'Zero-Waste Launch',
        'key_points': 'Natural ingredients, plastic-free'
    }, style="Casual with emojis")
    print(f"\nGenerated Pitch:\n{pitch}")
