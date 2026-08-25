from datetime import datetime
import requests
import pandas as pd
import os


API_KEY = "840d6e0581f570924c9dd797ad3191eb"

movie_ids = [
    550,278,238,680,13,
    155,424,122,603,27205,
    157336,11,120,121,122917,
    597,672,671,1891,24428
]

reviews = []

print("Collecting reviews from TMDB...")

for movie_id in movie_ids:

    url = (
        f"https://api.themoviedb.org/3/movie/"
        f"{movie_id}/reviews?api_key={API_KEY}"
    )

    response = requests.get(url)

    data = response.json()

    if "results" in data:

        for review in data["results"]:

            reviews.append({
    "movie_id": movie_id,
    "review": review["content"],
    "collection_time": datetime.now()
})

df = pd.DataFrame(reviews)

os.makedirs("external_data", exist_ok=True)

df.to_csv(
    "external_data/tmdb_reviews.csv",
    index=False
)

print(f"\nCollected {len(df)} reviews")
print("Saved to external_data/tmdb_reviews.csv")