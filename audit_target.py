import pandas as pd

devices = pd.read_csv("devices-1681209661.csv", usecols=["id"])
events = pd.read_csv("events-1681209680.csv", low_memory=False)

reason = events["reason"].fillna("").astype(str).str.lower()
cause = events["determined_cause"].fillna("").astype(str).str.lower()

tests = {
    "Explicit device failure": reason.str.contains(
        "device failure", regex=False
    ),
    "Failure or malfunction": reason.str.contains(
        r"failure|malfunction|defect|broken|breakage", regex=True
    ),
    "Leak or integrity issue": reason.str.contains(
        r"leak|integrity failure|compromised", regex=True
    ),
    "Device failure or malfunction cause": cause.str.contains(
        r"device design|nonconforming material|software design|"
        r"process control|equipment maintenance", regex=True
    )
}

print("\nTARGET DEFINITION AUDIT")
print("-" * 65)

for name, mask in tests.items():
    event_count = int(mask.sum())
    unique_devices = events.loc[mask, "device_id"].nunique()
    print(f"\n{name}")
    print("Matching event records:", event_count)
    print("Unique devices:", unique_devices)

combined = tests["Explicit device failure"] | tests["Failure or malfunction"]
positive_ids = set(events.loc[combined, "device_id"].dropna())
positive_count = devices["id"].isin(positive_ids).sum()

print("\nCOMBINED EXPLICIT FAILURE + MALFUNCTION")
print("Positive devices:", int(positive_count))
print("Negative devices:", len(devices) - int(positive_count))
print("Positive percentage:", round(positive_count / len(devices) * 100, 3))

print("\nEXPLICIT DEVICE FAILURE REASON VALUES")
print(
    events.loc[tests["Explicit device failure"], "reason"]
    .value_counts().head(20).to_string()
)
