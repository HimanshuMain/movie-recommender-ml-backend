import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load datasets ----------
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


movies = movies.merge(credits, on="title")

def parse_names(text, limit=None):
    try:
        data = ast.literal_eval(text)
        names = [i["name"] for i in data]
        return " ".join(names[:limit]) if limit else " ".join(names)
    except:
        return ""

def get_director(crew):
    try:
        crew = ast.literal_eval(crew)
        for person in crew:
            if person["job"] == "Director":
                return person["name"]
    except:
        pass
    return ""

# ---------- Feature Engineering --------
movies["genres"] = movies["genres"].apply(parse_names)
movies["keywords"] = movies["keywords"].apply(parse_names)
movies["cast"] = movies["cast"].apply(lambda x: parse_names(x, limit=5))
movies["director"] = movies["crew"].apply(get_director)

movies["combined_features"] = (
    movies["overview"].fillna("") + " " +
    movies["genres"] + " " +
    movies["keywords"] + " " +
    movies["cast"] + " " +
    movies["director"]
)

# ---------- Vectorization ----------
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
tfidf_matrix = vectorizer.fit_transform(movies["combined_features"])

similarity_matrix = cosine_similarity(tfidf_matrix)

# ---------- Hybrid Recommendation ----------
def recommend(title, top_n=25):
    if title not in movies["title"].values:
        return fallback_recommendations()

    idx = movies[movies["title"] == title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in scores[1:top_n + 1]:
        recommendations.append({
            "title": movies.iloc[i]["title"],
            "similarity": round(score * 100, 2)
        })

    return recommendations

def fallback_recommendations():
    sample = movies.sample(10)
    return [
        {"title": row["title"], "similarity": 50.0}
        for _, row in sample.iterrows()
    ]
