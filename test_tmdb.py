import requests

API_KEY = "840d6e0581f570924c9dd797ad3191eb"

url = f"https://api.themoviedb.org/3/movie/550/reviews?api_key={API_KEY}"

response = requests.get(url)

print("Status Code:", response.status_code)

data = response.json()

print(data.keys())