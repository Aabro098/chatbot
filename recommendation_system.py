import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from pydantic import BaseModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class SkillsRequest(BaseModel):
    user_skills: List[str]
    user_requested_skills: List[str] = []
    user_interactions: str = ""
    description: str = ""
    top_k: int = 10


# Setup Hugging Face hosted model
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_TOKEN")


def load_llm(huggingface_repo_id):
    """Load the Hugging Face chat model"""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set in the environment")
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.3,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=256,
    )
    return ChatHuggingFace(llm=llm)


def set_custom_prompt(custom_prompt_template):
    """Create a chat prompt template"""
    return ChatPromptTemplate.from_template(custom_prompt_template)


def filter_by_prerequisites(df, user_skills):
    """Filter skills based on user's prerequisite knowledge"""
    def prereq_ok(prereqs):
        if not prereqs:
            return True
        prereq_list = [p.strip().lower() for p in prereqs.split(",")]
        return any(skill.lower() in prereq_list for skill in user_skills)

    return df[df["prerequisites_text"].apply(prereq_ok)]


def load_recommendation_models():
    """Load saved artifacts for recommendation system"""
    df = pd.read_pickle("skills_with_embeddings.pkl")
    embeddings = np.vstack(df["embedding"].values)
    index = faiss.read_index("skills.index")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, index, model


def structured_score(row, similarity, match_score=0.0):
    """Calculate weighted score for skill recommendation"""
    sim_w = 0.45
    demand_w = 0.15
    future_w = 0.15
    match_w = 0.25
    return (
        sim_w * similarity +
        demand_w * (row["job_demand_score"] / 100) +
        future_w * (row["future_relevance_score"] / 100) +
        match_w * match_score
    )


def refine_skills_with_llm(user_interactions, description, recommended_skills):
    """Refine skill recommendations using a Hugging Face hosted model"""
    prompt = f"""You are an expert career advisor. Analyze the user profile and refine the skill recommendations.

User Description: {description}

User Interactions: {user_interactions}

Current Recommended Skills: {recommended_skills}

Return ONLY a JSON array of refined skill names. No explanation or extra text.
Example: ["skill1", "skill2", "skill3"]

JSON Array:"""
    
    try:
        print(f"Calling Hugging Face model: {HF_MODEL_ID}")
        llm = load_llm(HF_MODEL_ID)
        chain = set_custom_prompt(prompt) | llm | StrOutputParser()
        content = chain.invoke({})
        
        # Try to parse the JSON array from the response
        try:
            skills = json.loads(content)
            if isinstance(skills, list):
                return skills
        except Exception:
            # Fallback: try to extract JSON array from text
            import re
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                try:
                    skills = json.loads(match.group(0))
                    if isinstance(skills, list):
                        return skills
                except:
                    pass
        
        return recommended_skills
    except Exception as e:
        print(f"Error refining skills with Hugging Face: {type(e).__name__}: {str(e)}")
        print(f"Returning original recommendations as fallback")
        return recommended_skills


def recommend_skills(df, index, embeddings, user_skills, top_k=10):
    """Generate skill recommendations based on user skills"""
    user_set = set([s.strip().lower() for s in user_skills if s.strip()])
    mask = df['skill_name'].fillna('').str.lower().isin(user_set)
    if mask.any():
        user_query_emb = np.vstack(df.loc[mask, 'embedding'].values).mean(axis=0)
    else:
        user_query_emb = np.mean(embeddings, axis=0)
    
    scores, indices = index.search(user_query_emb.reshape(1, -1), k=200)
    candidates = df.iloc[indices[0]].copy().reset_index(drop=True)
    candidates["similarity"] = scores[0]
    
    def overlap_count(text):
        if not text or pd.isna(text):
            return 0
        toks = set([t.strip().lower() for t in text.split(",") if t.strip()])
        return len(user_set & toks)
    
    candidates["prereq_overlap"] = candidates["prerequisites_text"].fillna("").apply(overlap_count)
    candidates["comp_overlap"] = candidates["complementary_text"].fillna("").apply(overlap_count)
    candidates["skill_name_match"] = candidates["skill_name"].fillna("").apply(lambda s: 1 if s.strip().lower() in user_set else 0)
    
    denom = max(1, len(user_set))
    candidates["match_score"] = (
        0.6 * candidates["skill_name_match"] +
        0.3 * (candidates["prereq_overlap"] / denom) +
        0.1 * (candidates["comp_overlap"] / denom)
    )
    
    candidates["final_score"] = candidates.apply(
        lambda r: structured_score(r, r["similarity"], r["match_score"]),
        axis=1
    )
    
    # Filter by prerequisites
    filtered_candidates = filter_by_prerequisites(candidates, user_skills)
    result = filtered_candidates.sort_values("final_score", ascending=False).drop_duplicates(subset=["skill_name"]).head(top_k).reset_index(drop=True)
    return result
