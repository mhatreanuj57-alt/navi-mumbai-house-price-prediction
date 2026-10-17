import re
import pandas as pd

RAW = "data/mumbai1.csv"
OUT = "data/navimumbai_listings.csv"
MIN_ROWS_PER_NODE = 10

NODES_NM = ["kharghar", "vashi", "nerul", "seawoods", "belapur", "airoli",
            "ghansoli", "kopar khairane", "sanpada", "juinagar", "kamothe",
            "kalamboli", "panvel", "ulwe", "dronagiri", "taloja", "khandeshwar"]


def normalize_location(loc: str):
    s = str(loc).strip().lower()
    s = re.sub(r"^sector[\s\-\.]*\d+[a-z]?\s*", "", s)
    s = re.sub(r"\s*\(.*?\)", "", s)
    for node in sorted(NODES_NM, key=len, reverse=True):
        if node in s:
            return node.title()
    return None


def main():
    df = pd.read_csv(RAW)
    df = df.rename(columns={"sqrt": "sqft"})
    df["node"] = df["location"].apply(normalize_location)
    df = df.dropna(subset=["node"])

    df = df[(df["price"].between(800_000, 150_000_000)) & (df["sqft"].between(150, 5000))]
    df = df[df["bhk"].between(1, 5)]
    df["price_per_sqft"] = df["price"] / df["sqft"]
    df = df[df["price_per_sqft"].between(2000, 60_000)]

    amenity_cols = ["Gymnasium", "Lift Available", "CarParking", "Maintenance Staff",
                    "24x7 Security", "Children's Play Area", "Clubhouse", "Intercom",
                    "Landscaped Gardens", "Indoor Games", "Gas Connection",
                    "Jogging Track", "Swimming Pool"]
    df["amenities"] = df[amenity_cols].sum(axis=1)

    counts = df["node"].value_counts()
    keep = counts[counts >= MIN_ROWS_PER_NODE].index
    df = df[df["node"].isin(keep)].copy()

    df = df.drop_duplicates(subset=["price", "sqft", "location", "bhk"])
    df["is_new"] = (df["New/Resale"] == 1).astype(int)

    cols = ["price", "sqft", "bhk", "is_new", "amenities", "node", "location"]
    df[cols].to_csv(OUT, index=False)
    print(f"Saved {len(df)} listings across {df['node'].nunique()} nodes")
    print(df["node"].value_counts())
    print("\nMedian rate per sqft by node:")
    print((df.groupby("node")["price_per_sqft"].median() / 1000).round(1).sort_values(ascending=False))


if __name__ == "__main__":
    main()
