import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    "Title": [
        "Inception", "Interstellar", "The Prestige",
        "The Dark Knight", "Tenet", "Memento"
    ],
    "Genre": [
        "Sci-Fi Action", "Sci-Fi Drama", "Drama Mystery",
        "Action Crime", "Sci-Fi Action", "Thriller Mystery"
    ],
    "Plot": [
        "dream manipulation", "space exploration",
        "magicians rivalry", "batman joker",
        "time inversion", "memory loss"
    ]
}

df = pd.DataFrame(data)

df["features"] = df["Genre"] + " " + df["Plot"]

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform(df["features"])
similarity = cosine_similarity(vectors)

def recommend(movie_title):
    if movie_title not in df["Title"].values:
        return []

    idx = df[df["Title"] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = [df.iloc[i[0]]["Title"] for i in scores[1:6]]
    return recommendations
