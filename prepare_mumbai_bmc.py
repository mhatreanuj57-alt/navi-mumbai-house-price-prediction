"""Create a clean, BMC-only training dataset from the sourced Mumbai listings."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "mumbai_bmc_source" / "mumbai-house-price-data-cleaned.csv"
OUT = ROOT / "data" / "mumbai_bmc_listings.csv"
MIN_LISTINGS_PER_LOCALITY = 50

# Localities administered by BMC.  This deliberately excludes Thane, Mira-Bhayandar,
# Navi Mumbai, Kalyan-Dombivli, Vasai-Virar, and other MMR municipal corporations.
BMC_LOCALITIES = {
    "Agripada", "Andheri", "Bandra", "Bhandup", "Borivali", "Byculla", "Chembur",
    "Chinchpokli", "Colaba", "Cuffe Parade", "Dadar", "Dahisar", "Deonar", "Fort",
    "Ghatkopar", "Girgaon", "Goregaon", "Jogeshwari", "Juhu", "Kandivali", "Kanjurmarg",
    "Khar", "Kurla", "Lower Parel", "Mahalaxmi", "Mahim", "Malabar Hill", "Malad",
    "Matunga", "Mazagaon", "Mulund", "Parel", "Powai", "Prabhadevi", "Santacruz",
    "Sewri", "Sion", "Tardeo", "Vikhroli", "Vile Parle", "Ville Parle", "Wadala", "Worli",
}


def normalise_locality(value: str) -> str:
    value = str(value).strip()
    aliases = {"Ville Parle": "Vile Parle", "Vikroli": "Vikhroli", "Mazgaon": "Mazagaon"}
    return aliases.get(value, value)


def ward_for(locality: str) -> str:
    """Human-readable BMC ward group shown in the product; not a claimed official ward ID."""
    south = {"Colaba", "Cuffe Parade", "Fort", "Girgaon", "Malabar Hill", "Tardeo", "Mahalaxmi", "Byculla", "Mazagaon", "Agripada", "Chinchpokli", "Lower Parel", "Parel", "Prabhadevi", "Worli", "Sewri"}
    western = {"Bandra", "Khar", "Santacruz", "Vile Parle", "Juhu", "Andheri", "Jogeshwari", "Goregaon", "Malad", "Kandivali", "Borivali", "Dahisar"}
    central = {"Dadar", "Mahim", "Matunga", "Wadala", "Sion", "Chembur", "Kurla", "Ghatkopar", "Bhandup", "Mulund", "Vikhroli", "Kanjurmarg", "Powai", "Deonar"}
    if locality in south:
        return "Mumbai City"
    if locality in western:
        return "Western Suburbs"
    if locality in central:
        return "Eastern / Central Suburbs"
    return "Mumbai"


def main() -> None:
    df = pd.read_csv(RAW)
    df["node"] = df["locality"].map(normalise_locality)
    df = df[df["node"].isin(BMC_LOCALITIES)].copy()
    df = df[df["price"].between(800_000, 200_000_000) & df["area"].between(150, 5_000)]
    df = df[df["bedroom_num"].between(1, 5)]
    df["price_per_sqft"] = df["price"] / df["area"]
    df = df[df["price_per_sqft"].between(3_000, 150_000)]
    df["sqft"] = df["area"]
    df["bhk"] = df["bedroom_num"]
    df["age_years"] = df["age"].clip(lower=0, upper=100)
    df["is_new"] = (df["age_years"] <= 1).astype(int)
    df["bmc_ward"] = df["node"].map(ward_for)
    df["region"] = "Mumbai / BMC"

    counts = df["node"].value_counts()
    df = df[df["node"].isin(counts[counts >= MIN_LISTINGS_PER_LOCALITY].index)]
    cols = ["price", "sqft", "bhk", "is_new", "age_years", "node", "bmc_ward", "region", "latitude", "longitude"]
    output = df[cols].drop_duplicates()
    output.to_csv(OUT, index=False)
    print(f"Saved {len(output):,} BMC listings across {output['node'].nunique()} localities -> {OUT}")
    print(df["node"].value_counts().to_string())


if __name__ == "__main__":
    main()
