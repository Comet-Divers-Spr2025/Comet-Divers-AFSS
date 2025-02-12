import requests
import pandas as pd

# Query API: https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html
# Query filter: https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html

SBDB_QUERY = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"


def get_comets():
    fields = [
        "full_name",
        "spkid",
        # Elements
        "epoch",  # in JD
        "e",  # eccentricity
        "a",  # semimajor axis (au)
        "q",  # perihelion (au)
        "i",  # inclination (deg)
        "om",  # longitude of ascending node (deg)
        "w",  # argument of perihelion (deg)
        "ma",  # mean anomaly (deg)
        "tp",  # time of perihelion (JD)
        # Uncertainties
        "sigma_e",
        "sigma_a",
        "sigma_q",
        "sigma_i",
        "sigma_om",
        "sigma_w",
        "sigma_tp",
        "sigma_ma",
        # Observation parameters
        "data_arc",  # number of days of observation
        "first_obs",
        "last_obs",
        "n_obs_used",
    ]

    params = {
        "sb-kind": "c",  # Comets only
        # "sb-class": "PAR,HYP",  # Hyperbolic and parabolic comets only
        "full-prec": "true",  # Full numerical precision
        "fields": ",".join(fields),
    }

    resp = requests.get(SBDB_QUERY, params=params).json()

    # Fail if API changes
    assert resp["signature"]["version"] == "1.0"

    df = pd.DataFrame(resp["data"], columns=resp["fields"])

    # Clean up name field
    df["full_name"] = df["full_name"].map(lambda x: x.strip())

    df["q"] = df["q"].astype(pd.Float64Dtype())
    df["tp"] = df["tp"].astype(pd.Float64Dtype())
    df["e"] = df["e"].astype(pd.Float64Dtype())

    return df


def filter_comets(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to 0.5-1.5 AU
    df = df[df["q"] > 0.5]
    df = df[df["q"] < 1.5]

    # Filter to >0.95 ecc
    df = df[df["e"] > 0.95]

    # Filter to comets with perihelion after 1970
    df = df[df["tp"] > 2440605.5]

    return df


if __name__ == "__main__":
    df = get_comets()
    df = filter_comets(df)

    print(f"Got {len(df)} comets")

    df.to_csv("comets.csv", index=False)
