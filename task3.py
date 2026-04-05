# ==============================
# RESUME SCREENING SYSTEM + GRAPH
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

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
TEXT_COL = "job_title"

print("\nUsing Text Column:", TEXT_COL)

# ==============================
# 3. CLEAN DATA
# ==============================
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

# ==============================
# 4. JOB DESCRIPTION
# ==============================
job_description = """
Looking for a Data Scientist with Python, Machine Learning,
Data Analysis, and Visualization skills.
"""

# ==============================
# 5. TF-IDF VECTORIZATION
# ==============================
vectorizer = TfidfVectorizer()

all_text = df[TEXT_COL].tolist()
all_text.append(job_description)

X = vectorizer.fit_transform(all_text)

job_vector = X[-1]
resume_vectors = X[:-1]

# ==============================
# 6. SIMILARITY SCORING
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
# 8. SKILL GAP ANALYSIS
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
# 9. BEST CANDIDATE
# ==============================
best = df_ranked.iloc[0]

print("\nBest Candidate:")
print(best[['candidate_id', 'job_title', 'Score']])

# ==============================
# 10. GRAPH (IMPORTANT 🔥)
# ==============================

# Top 10 Candidates Graph
top10 = df_ranked.head(10)

plt.figure()
plt.bar(top10['candidate_id'].astype(str), top10['Score'])

plt.title("Top 10 Candidate Scores")
plt.xlabel("Candidate ID")
plt.ylabel("Similarity Score")

plt.xticks(rotation=45)
plt.show()