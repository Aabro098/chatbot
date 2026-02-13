import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import faiss

# Load data
df = pd.read_csv("skills_dataset.csv")

def collect_columns(prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return df[cols].fillna("").agg(", ".join, axis=1)

df["prerequisites_text"] = collect_columns("prerequisites/")
df["complementary_text"] = collect_columns("complementary_skills/")
df["industry_text"] = collect_columns("industry_usage/")

# Load or fine-tune model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Fine-tuning configuration
do_finetune = True  # Set to True to enable fine-tuning
finetune_epochs = 30
finetune_batch_size = 16
save_finetuned_model = True
finetuned_model_path = "fine_tuned_sbert"

if do_finetune:
    examples = []
    for _, r in df.iterrows():
        skill_text = f"Skill: {r['skill_name']} Category: {r['category']} Industry: {r['industry_text']}"
        pos = r['complementary_text'] if pd.notna(r['complementary_text']) and str(r['complementary_text']).strip() else r['prerequisites_text']
        if pd.isna(pos) or not str(pos).strip():
            continue
        pos_text = f"Context: {pos}"
        examples.append(InputExample(texts=[skill_text, pos_text]))
    if len(examples) == 0:
        print("No training examples found; skipping fine-tune")
    else:
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=finetune_batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)
        warmup_steps = max(100, int(len(train_dataloader) * finetune_epochs * 0.1))
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=finetune_epochs, warmup_steps=warmup_steps)
        if save_finetuned_model:
            model.save(finetuned_model_path)

def build_weighted_embedding(row):
    skill_emb = model.encode(
        f"Skill: {row['skill_name']}",
        normalize_embeddings=True
    )
    context_emb = model.encode(
        f"""Category: {row['category']}
Industry Usage: {row['industry_text']}
Complementary Skills: {row['complementary_text']}""",
        normalize_embeddings=True
    )
    market_emb = model.encode(
        f"""Job Demand Score: {row['job_demand_score']}
Future Relevance Score: {row['future_relevance_score']}
Market Trend: {row['market_trend']}""",
        normalize_embeddings=True
    )
    return 0.5 * skill_emb + 0.3 * context_emb + 0.2 * market_emb

df["embedding"] = df.apply(build_weighted_embedding, axis=1)
embeddings = np.vstack(df["embedding"].values)

# Build and save FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
faiss.write_index(index, "skills.index")

# Save DataFrame (with embeddings)
df.to_pickle("skills_with_embeddings.pkl")

# Save model (if fine-tuned)
# model.save("fine_tuned_sbert")