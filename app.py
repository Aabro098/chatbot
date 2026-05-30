import os
import uvicorn
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Import modules
from recommendation_system import (
    SkillsRequest,
    load_recommendation_models,
    recommend_skills as recommend_skills_func,
    refine_skills_with_llm
)
from chatbot_api import (
    QueryRequest,
    QueryResponse,
    process_query
)

# Initialize FastAPI app
app = FastAPI(
    title="SkillSwap API",
    description="API for querying documents using RAG with LLM and recommending skills based on user input.",
    version="1.0.0"
)

# Load recommendation system models globally
print("Loading recommendation system models...")
df, embeddings, index, model = load_recommendation_models()
print("✓ Recommendation system models loaded")

print("Loading chatbot API with Hugging Face hosted model...")
# Chatbot chain is loaded in chatbot_api.py
print(f"✓ Chatbot API loaded (using HF_MODEL_ID from environment)")


# ==================== HEALTH CHECK ENDPOINTS ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - Health check"""
    return {"message": "SkillSwap API is running", "status": "healthy"}


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}


# ==================== RECOMMENDATION SYSTEM ENDPOINTS ====================

@app.post("/recommend", tags=["Recommendations"])
async def recommend_skills_api(request: SkillsRequest):
    """
    Get skill recommendations based on user's current skills
    
    Parameters:
    - user_skills: List of skills the user currently has
    - user_requested_skills: Optional list of skills user is interested in
    - user_interactions: Optional user interaction history
    - description: Optional user description
    - top_k: Number of top recommendations to return (default: 10)
    
    Returns:
    - recommended_skills: List of recommended skills
    """
    print(f"Received recommendation request with user skills: {request.user_skills}")
    print(f"Additional context - Requested skills: {request.user_requested_skills}")
    print(f"Additional context - User interactions: {request.user_interactions}")
    print(f"Additional context - Description: {request.description}")
    
    try:
        combined_skills = list(set(request.user_skills + request.user_requested_skills))
        
        # Get initial recommendations
        result = recommend_skills_func(
            df,
            index,
            embeddings,
            combined_skills,
            top_k=request.top_k,
            description=request.description,
        )
        recommended_skills = result["skill_name"].tolist()
        
        # Refine with hosted LLM if description or interactions are provided
        if request.description or request.user_interactions:
            new_skills = refine_skills_with_llm(
                user_interactions=request.user_interactions,
                description=request.description,
                recommended_skills=recommended_skills
            )
            return {"recommended_skills": new_skills}
        else:
            return {"recommended_skills": recommended_skills}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


# ==================== CHATBOT ENDPOINTS ====================

@app.post("/query", response_model=QueryResponse, tags=["Chat"])
async def query(request: QueryRequest):
    """
    Send a query to the chatbot and get a response
    
    Parameters:
    - query: The question to ask the chatbot
    
    Returns:
    - query: The original query
    - response: The chatbot's response
    """
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        response = process_query(request.query)
        return QueryResponse(query=request.query, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ==================== MAIN ====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print(f"\n🚀 Starting SkillSwap API on port {port}...")
    print(f"🤖 System Configuration:")
    print(f"   - Recommendations: Hugging Face hosted model")
    print(f"   - Chatbot: Hugging Face hosted model")
    print(f"📚 Endpoints available:")
    print(f"   - POST /recommend - Get skill recommendations")
    print(f"   - POST /query - Chat with the bot")
    print(f"   - GET / - Health check")
    print(f"\n⚠️  Make sure HF_TOKEN is set in your environment\n")
    print(f"API Docs: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
