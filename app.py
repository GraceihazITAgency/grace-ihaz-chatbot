"""
Grace-Ihaz Properties AI Chatbot Backend - Production Version
Handles all edge cases, token limits, and error scenarios
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import joblib
import numpy as np
import pandas as pd
import httpx
import json
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Grace-Ihaz Property Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD ML MODEL
# ============================================================================
try:
    model = joblib.load('property_price_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    encoders = joblib.load('label_encoders.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    logger.info("✓ ML model loaded")
    
    VALID_STATES = list(encoders['state'].classes_)
    VALID_CITIES = list(encoders['city'].classes_)
    VALID_PROPERTY_TYPES = list(encoders['property_type'].classes_)
    VALID_CONDITIONS = list(encoders['condition'].classes_)
    VALID_FURNISHED = list(encoders['furnished'].classes_)
    
except Exception as e:
    logger.error(f"Model load error: {e}")
    model = scaler = encoders = None
    VALID_STATES = VALID_CITIES = VALID_PROPERTY_TYPES = VALID_CONDITIONS = VALID_FURNISHED = []

try:
    df_properties = pd.read_csv('grace_ihaz_property_dataset.csv')
    logger.info(f"✓ Loaded {len(df_properties)} properties")
    
    # Get available states and cities from dataset
    AVAILABLE_STATES = sorted(df_properties['state'].unique().tolist())
    AVAILABLE_CITIES = sorted(df_properties['city'].unique().tolist())
    AVAILABLE_PROPERTY_TYPES = sorted(df_properties['property_type'].unique().tolist())
    
except Exception as e:
    logger.error(f"Dataset load error: {e}")
    df_properties = None
    AVAILABLE_STATES = AVAILABLE_CITIES = AVAILABLE_PROPERTY_TYPES = []

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    conversation_history: Optional[List[Dict]] = []

# ============================================================================
# GROQ AI CLIENT
# ============================================================================
class GroqAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = "llama-3.3-70b-versatile"
        self.max_tokens = 1500  # Reduced to prevent token limit issues
    
    def truncate_messages(self, messages: List[Dict], max_messages: int = 8) -> List[Dict]:
        """Keep only recent messages to avoid token limits"""
        if len(messages) <= max_messages:
            return messages
        
        # Always keep system message + recent messages
        system_msgs = [m for m in messages if m.get('role') == 'system']
        other_msgs = [m for m in messages if m.get('role') != 'system']
        
        # Keep last N messages
        recent_msgs = other_msgs[-(max_messages-1):]
        
        return system_msgs + recent_msgs
    
    async def chat_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None):
        """Send chat completion with error handling"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Truncate message history to prevent token limits
        messages = self.truncate_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": self.max_tokens,
            "top_p": 1,
            "stream": False
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 400:
                    # Token limit or bad request - retry without tools
                    logger.warning("400 error, retrying without tools")
                    payload.pop("tools", None)
                    payload.pop("tool_choice", None)
                    payload["max_tokens"] = 800
                    
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                logger.error(f"Groq API error: {e}")
                # Return fallback response instead of crashing
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "I apologize, I'm experiencing technical difficulties. Please try asking a simpler question or search for properties directly."
                        }
                    }]
                }

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')
groq_client = GroqAIClient(GROQ_API_KEY)

# ============================================================================
# SIMPLIFIED SYSTEM PROMPT
# ============================================================================
SYSTEM_PROMPT = f"""You are Grace, AI assistant for Grace-Ihaz Properties in Nigeria.

Available locations: {', '.join(AVAILABLE_STATES[:5])}
Property types: {', '.join(AVAILABLE_PROPERTY_TYPES[:5])}

Your role:
- Search properties using search_properties function
- Be brief and helpful
- If you can't answer, offer alternatives

Keep responses under 100 words."""

# ============================================================================
# SIMPLIFIED TOOLS
# ============================================================================
CHATBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": "Search properties by location, type, bedrooms, price",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string"},
                    "city": {"type": "string"},
                    "property_type": {"type": "string"},
                    "min_bedrooms": {"type": "integer"},
                    "max_bedrooms": {"type": "integer"},
                    "min_price": {"type": "number"},
                    "max_price": {"type": "number"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_stats",
            "description": "Get average prices and market info",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {"type": "string"}
                }
            }
        }
    }
]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def search_properties_db(criteria: dict) -> List[dict]:
    """Search properties"""
    try:
        df = df_properties.copy()
        
        if criteria.get('state'):
            df = df[df['state'].str.contains(criteria['state'], case=False, na=False)]
        if criteria.get('city'):
            df = df[df['city'].str.contains(criteria['city'], case=False, na=False)]
        if criteria.get('property_type'):
            df = df[df['property_type'].str.contains(criteria['property_type'], case=False, na=False)]
        if criteria.get('min_bedrooms'):
            df = df[df['bedrooms'] >= criteria['min_bedrooms']]
        if criteria.get('max_bedrooms'):
            df = df[df['bedrooms'] <= criteria['max_bedrooms']]
        if criteria.get('min_price'):
            df = df[df['price_ngn'] >= criteria['min_price']]
        if criteria.get('max_price'):
            df = df[df['price_ngn'] <= criteria['max_price']]
        
        results = df.head(5).to_dict('records')
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def get_market_stats(state: str = None) -> dict:
    """Get market statistics"""
    try:
        df = df_properties[df_properties['listing_type'] == 'Sale'].copy()
        
        if state:
            df = df[df['state'].str.contains(state, case=False, na=False)]
        
        if len(df) == 0:
            return {"error": "No data available"}
        
        return {
            "total_properties": int(len(df)),
            "average_price": float(df['price_ngn'].mean()),
            "median_price": float(df['price_ngn'].median()),
            "min_price": float(df['price_ngn'].min()),
            "max_price": float(df['price_ngn'].max())
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"error": str(e)}

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute tool"""
    try:
        if tool_name == "search_properties":
            properties = search_properties_db(arguments)
            if properties:
                return json.dumps({
                    "count": len(properties),
                    "properties": properties,
                    "message": f"Found {len(properties)} properties"
                })
            else:
                return json.dumps({
                    "count": 0,
                    "properties": [],
                    "message": "No properties found. Try different criteria."
                })
        
        elif tool_name == "get_market_stats":
            stats = get_market_stats(arguments.get('state'))
            return json.dumps({
                "statistics": stats,
                "message": f"Market data for {arguments.get('state', 'Nigeria')}"
            })
        
        return json.dumps({"error": "Unknown tool"})
    
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return json.dumps({"error": str(e)})

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    return {
        "service": "Grace-Ihaz Property Chatbot",
        "status": "online",
        "version": "3.0 (Production)",
        "available_states": AVAILABLE_STATES,
        "total_properties": len(df_properties) if df_properties is not None else 0
    }

@app.post("/chat")
async def chat(request: ChatMessage):
    """Main chat endpoint with comprehensive error handling"""
    try:
        # Build messages with truncation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Only keep last 5 conversation turns to prevent token issues
        history = request.conversation_history[-10:] if request.conversation_history else []
        messages.extend(history)
        messages.append({"role": "user", "content": request.message})
        
        # Get AI response
        response = await groq_client.chat_completion(messages, CHATBOT_TOOLS)
        
        assistant_message = response['choices'][0]['message']
        
        # Handle tool calls
        if assistant_message.get('tool_calls'):
            try:
                tool_call = assistant_message['tool_calls'][0]
                function_name = tool_call['function']['name']
                function_args = json.loads(tool_call['function']['arguments'])
                
                logger.info(f"Tool: {function_name} | Args: {function_args}")
                
                tool_result = execute_tool(function_name, function_args)
                
                # Get final response
                messages.append(assistant_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "name": function_name,
                    "content": tool_result
                })
                
                # Truncate before final call
                messages = groq_client.truncate_messages(messages, max_messages=6)
                
                final_response = await groq_client.chat_completion(messages, CHATBOT_TOOLS)
                final_content = final_response['choices'][0]['message'].get('content', 
                    'Here are the results I found for you.')
                
                return {
                    "response": final_content,
                    "function_called": function_name,
                    "function_result": json.loads(tool_result),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as tool_error:
                logger.error(f"Tool execution error: {tool_error}")
                # Fallback: return properties without AI summary
                return {
                    "response": "I found some properties for you. Please review the results below.",
                    "function_called": function_name,
                    "function_result": json.loads(tool_result) if 'tool_result' in locals() else {},
                    "timestamp": datetime.now().isoformat()
                }
        
        # No tool call - direct response
        content = assistant_message.get('content', 'How can I help you with properties today?')
        return {
            "response": content,
            "function_called": None,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Graceful fallback
        return {
            "response": "I'm here to help! You can ask me to search for properties in Lagos, Abuja, Rivers, Oyo, or Kano. What are you looking for?",
            "error": "fallback_response",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/search")
async def search_direct(state: Optional[str] = None, city: Optional[str] = None, 
                       property_type: Optional[str] = None, min_price: Optional[float] = None,
                       max_price: Optional[float] = None):
    """Direct property search without AI"""
    criteria = {k: v for k, v in {
        'state': state, 'city': city, 'property_type': property_type,
        'min_price': min_price, 'max_price': max_price
    }.items() if v is not None}
    
    properties = search_properties_db(criteria)
    return {
        "count": len(properties),
        "properties": properties
    }

@app.get("/locations")
async def get_locations():
    """Get available locations"""
    return {
        "states": AVAILABLE_STATES,
        "cities": AVAILABLE_CITIES[:20],  # Top 20 cities
        "property_types": AVAILABLE_PROPERTY_TYPES
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  GRACE-IHAZ PROPERTIES AI CHATBOT - PRODUCTION v3.0")
    print("="*70)
    print(f"\n✓ Properties: {len(df_properties) if df_properties is not None else 0}")
    print(f"✓ States: {', '.join(AVAILABLE_STATES)}")
    print(f"✓ API: http://localhost:8000")
    print(f"✓ Docs: http://localhost:8000/docs")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)