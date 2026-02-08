import csv

csv_path = "withoutmask_generated_embed.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["name"] + [f"emb_{i}" for i in range(1, 513)]
    writer.writerow(header)
