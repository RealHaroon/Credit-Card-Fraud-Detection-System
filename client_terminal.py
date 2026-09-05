# get_sample_transactions.py
import pandas as pd
import json

df = pd.read_csv("/run/media/haroon/Local Disk (D:)/Datasets/creditcard.csv")

# one real fraud case, one real legitimate case
fraud_row = df[df["Class"] == 1].iloc[0].drop("Class").to_dict()
legit_row = df[df["Class"] == 0].iloc[0].drop("Class").to_dict()

print("=== FRAUD EXAMPLE (ground truth: Class=1) ===")
print(json.dumps(fraud_row, indent=2))

print("\n=== LEGITIMATE EXAMPLE (ground truth: Class=0) ===")
print(json.dumps(legit_row, indent=2))

# save both to files so you can just copy-paste into the web UI
with open("sample_fraud.json", "w") as f:
    json.dump(fraud_row, f, indent=2)
with open("sample_legit.json", "w") as f:
    json.dump(legit_row, f, indent=2)

print("\nSaved to sample_fraud.json and sample_legit.json")