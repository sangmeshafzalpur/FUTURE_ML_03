# ==============================
# RESUME SCREENING SYSTEM (FIXED)
# ==============================

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("resume.csv")

print("\nDataset Preview:")
print(df.head())

print("\nColumns:")
print(df.columns)

# ==============================
# 2. USE CORRECT TEXT COLUMN
# ==============================
TEXT_COL = "job_title"   # correct column from your dataset

print("\nUsing Text Column:", TEXT_COL)

# ==============================
# 3. CLEAN DATA
# ==============================
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

# ==============================
# 4. CREATE JOB DESCRIPTION
# ==============================
job_description = """
Looking for a Data Scientist with Python, Machine Learning,
Data Analysis, and Visualization skills.
"""

# ==============================
# 5. VECTORIZE TEXT
# ==============================
vectorizer = TfidfVectorizer()

all_text = df[TEXT_COL].tolist()
all_text.append(job_description)

X = vectorizer.fit_transform(all_text)

# Last item = job description
job_vector = X[-1]
resume_vectors = X[:-1]

# ==============================
# 6. SIMILARITY (SCORING)
# ==============================
scores = cosine_similarity(job_vector, resume_vectors).flatten()

df['Score'] = scores

# ==============================
# 7. RANKING
# ==============================
df_ranked = df.sort_values(by='Score', ascending=False)

print("\nTop Candidates:")
print(df_ranked[['candidate_id', 'job_title', 'Score']].head())

# ==============================
# 8. SKILL GAP (BASIC LOGIC)
# ==============================
required_skills = set([
    "python", "machine", "learning", "data",
    "analysis", "visualization", "sql"
])

def find_missing_skills(text):
    words = set(text.lower().split())
    return list(required_skills - words)

df['Missing Skills'] = df[TEXT_COL].apply(find_missing_skills)

print("\nSkill Gaps:")
print(df[['candidate_id', 'Missing Skills']].head())

# ==============================
# 9. OUTPUT BEST CANDIDATE
# ==============================
best = df_ranked.iloc[0]

print("\nBest Candidate:")
print(best[['candidate_id', 'job_title', 'Score']])