import pandas as pd
import hashlib
import json
import os

TRACK_FILE = "models/data_signature.json"

# Load reviews
df = pd.read_csv(
    "external_data/tmdb_reviews.csv"
)

# Create one big string from reviews
all_reviews = "".join(
    df["review"].astype(str).tolist()
)

# Generate hash
current_hash = hashlib.md5(
    all_reviews.encode("utf-8")
).hexdigest()

# First run
if not os.path.exists(TRACK_FILE):

    with open(
        TRACK_FILE,
        "w"
    ) as file:

        json.dump(
            {"hash": current_hash},
            file,
            indent=4
        )

    print("FIRST_RUN")

else:

    with open(
        TRACK_FILE,
        "r"
    ) as file:

        data = json.load(file)

    previous_hash = data["hash"]

    if current_hash == previous_hash:

        print("NO_NEW_DATA")

    else:

        print("NEW_DATA")

        with open(
            TRACK_FILE,
            "w"
        ) as file:

            json.dump(
                {"hash": current_hash},
                file,
                indent=4
            )