"""
PUCIT 403 - AI & Expert Systems assignment
Synthetic dataset of 5,000 patient records (focus: sickle cell traits/disease).

Steps:
 1. Generate 5,000 synthetic patients (Ghanaian context).
 2. Export the dataset to an Excel workbook.
 3. Identify all pairs of numerical variables.
 4. Generate a scatter plot for every numerical pair (saved as PNG and embedded in the workbook).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N = 5000
OUT_DIR = Path(__file__).parent
PLOT_DIR = OUT_DIR / "scatter_plots"
PLOT_DIR.mkdir(exist_ok=True)
XLSX = OUT_DIR / "synthetic_patient_records.xlsx"

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- categorical variables
gender = rng.choice(["Male", "Female"], size=N, p=[0.49, 0.51])

ethnicity = rng.choice(
    ["Akan", "Mole-Dagbani", "Ewe", "Ga-Adangbe", "Gurma", "Guan", "Grusi", "Mande", "Other"],
    size=N, p=[0.45, 0.18, 0.13, 0.07, 0.06, 0.04, 0.03, 0.01, 0.03])

blood_group = rng.choice(
    ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"],
    size=N, p=[0.49, 0.22, 0.20, 0.04, 0.025, 0.012, 0.010, 0.003])

# Haemoglobin genotype. In Ghana roughly 2 in 10 people carry the sickle (S) gene and
# a smaller share carry haemoglobin C; HbSS and HbSC are the main sickle cell disease forms.
genotypes = ["AA", "AS", "AC", "SS", "SC", "CC"]
genotype = rng.choice(genotypes, size=N, p=[0.715, 0.190, 0.065, 0.015, 0.012, 0.003])

status_map = {"AA": "Normal", "AC": "HbC trait", "CC": "HbC disease",
              "AS": "Sickle cell trait (carrier)",
              "SS": "Sickle cell disease (HbSS)", "SC": "Sickle cell disease (HbSC)"}
sickle_status = np.array([status_map[g] for g in genotype])

# ---------------------------------------------------------------- numerical variables
# Age: people with sickle cell disease have reduced life expectancy, so their ages skew younger.
age = rng.gamma(shape=4.0, scale=9.0, size=N) + 1
scd = np.isin(genotype, ["SS", "SC"])
age[scd] = rng.gamma(shape=3.0, scale=7.0, size=scd.sum()) + 1
too_old = age > 90
age[too_old] = rng.uniform(60, 90, size=too_old.sum())  # avoid a pile-up at the cap
age = np.round(age).astype(int)

# Height (cm): grows with age until about 18, sex difference in adults, SCD slightly shorter.
adult_height = np.where(gender == "Male", rng.normal(171, 7, N), rng.normal(159, 6.5, N))
growth = np.clip((age - 1) / 17, 0, 1)
height = 75 + (adult_height - 75) * growth ** 0.8
height = np.where(scd, height * 0.97, height)
height = np.round(height, 1)

# BMI: rises from childhood into middle age; SCD patients tend to be leaner.
bmi = np.where(age < 18, rng.normal(16.5, 2.0, N),
               rng.normal(22 + 0.08 * np.clip(age - 18, 0, 40), 4.0, N))
bmi = np.where(gender == "Female", bmi + rng.normal(0.8, 0.5, N), bmi)
bmi = np.where(scd, bmi - 2.5, bmi)
bmi = np.round(np.clip(bmi, 12.0, 45.0), 1)

# Weight (kg) derived from BMI and height so the three stay consistent: BMI = kg / m^2
weight = np.round(bmi * (height / 100) ** 2, 1)

# Haemoglobin (g/dL): strongly lowered by sickle cell disease.
hb_mean = {"AA": 13.2, "AS": 13.0, "AC": 12.9, "CC": 11.5, "SC": 10.8, "SS": 8.0}
hb_sd = {"AA": 1.3, "AS": 1.3, "AC": 1.3, "CC": 1.2, "SC": 1.1, "SS": 1.1}
hb = np.array([rng.normal(hb_mean[g], hb_sd[g]) for g in genotype])
hb = np.where((gender == "Male") & (age >= 15), hb + 1.0, hb)
hb = np.where(age < 5, hb - 1.0, hb)
hb = np.round(np.clip(hb, 4.5, 18.5), 1)

# Education: depends on age (young children cannot have completed university).
levels = ["None", "Primary", "JHS", "SHS", "Tertiary", "Postgraduate"]


def education_for(a):
    if a < 6:
        return "None"
    if a < 12:
        return "Primary"
    if a < 15:
        return rng.choice(["Primary", "JHS"], p=[0.3, 0.7])
    if a < 19:
        return rng.choice(["JHS", "SHS"], p=[0.4, 0.6])
    if a < 23:
        return rng.choice(["JHS", "SHS", "Tertiary"], p=[0.25, 0.45, 0.30])
    return rng.choice(levels, p=[0.12, 0.15, 0.23, 0.27, 0.18, 0.05])


education = np.array([education_for(a) for a in age])

df = pd.DataFrame({
    "Patient_ID": [f"PT{i:05d}" for i in range(1, N + 1)],
    "Age": age,
    "Gender": gender,
    "Height_cm": height,
    "Weight_kg": weight,
    "BMI": bmi,
    "Haemoglobin_g_dL": hb,
    "Blood_Group": blood_group,
    "Genotype": genotype,
    "Sickle_Cell_Status": sickle_status,
    "Ethnicity": ethnicity,
    "Education": education,
})

# ---------------------------------------------------------------- numerical pairs
numeric_cols = df.select_dtypes(include="number").columns.tolist()
pairs = list(combinations(numeric_cols, 2))
print("Numerical variables:", numeric_cols)
print(f"{len(pairs)} numerical pairs:")

colors = {"AA": "#4C78A8", "AS": "#F58518", "AC": "#54A24B",
          "SS": "#E45756", "SC": "#B279A2", "CC": "#72B7B2"}
pair_rows = []
plot_files = []
for i, (x, y) in enumerate(pairs, 1):
    r = df[x].corr(df[y])
    pair_rows.append({"Pair_No": i, "Variable_X": x, "Variable_Y": y, "Pearson_r": round(r, 3)})
    print(f"  {i:2d}. {x} vs {y}  (r = {r:.3f})")

    fig, ax = plt.subplots(figsize=(7, 5))
    for g in genotypes:
        sub = df[df["Genotype"] == g]
        ax.scatter(sub[x], sub[y], s=8, alpha=0.5, color=colors[g], label=f"Hb{g}")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(f"{x} vs {y}  (r = {r:.2f}, n = {N})")
    ax.legend(title="Genotype", markerscale=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f = PLOT_DIR / f"{i:02d}_{x}_vs_{y}.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    plot_files.append(f)

pairs_df = pd.DataFrame(pair_rows)

# ---------------------------------------------------------------- export to Excel
dictionary = pd.DataFrame([
    ("Patient_ID", "Text", "Unique synthetic identifier"),
    ("Age", "Numerical", "Age in years (1-95)"),
    ("Gender", "Categorical", "Male / Female"),
    ("Height_cm", "Numerical", "Height in centimetres"),
    ("Weight_kg", "Numerical", "Weight in kilograms (= BMI x height^2)"),
    ("BMI", "Numerical", "Body Mass Index, kg/m^2"),
    ("Haemoglobin_g_dL", "Numerical", "Haemoglobin level, g/dL"),
    ("Blood_Group", "Categorical", "ABO and Rh blood group"),
    ("Genotype", "Categorical", "Haemoglobin genotype: AA, AS, AC, SS, SC, CC"),
    ("Sickle_Cell_Status", "Categorical", "Interpretation of genotype"),
    ("Ethnicity", "Categorical", "Major Ghanaian ethnic groups"),
    ("Education", "Categorical (ordinal)", "Highest level: None, Primary, JHS, SHS, Tertiary, Postgraduate"),
], columns=["Variable", "Type", "Description"])

genotype_summary = (df.groupby(["Genotype", "Sickle_Cell_Status"])
                      .agg(Count=("Patient_ID", "count"),
                           Mean_Age=("Age", "mean"),
                           Mean_BMI=("BMI", "mean"),
                           Mean_Hb=("Haemoglobin_g_dL", "mean"))
                      .round(2).reset_index())
genotype_summary["Percent"] = (genotype_summary["Count"] / N * 100).round(2)

with pd.ExcelWriter(XLSX, engine="xlsxwriter") as xw:
    df.to_excel(xw, sheet_name="Patient_Records", index=False)
    dictionary.to_excel(xw, sheet_name="Data_Dictionary", index=False)
    df[numeric_cols].describe().round(2).to_excel(xw, sheet_name="Numeric_Summary")
    genotype_summary.to_excel(xw, sheet_name="Genotype_Summary", index=False)
    pairs_df.to_excel(xw, sheet_name="Numerical_Pairs", index=False)

    ws = xw.book.add_worksheet("Scatter_Plots")
    for k, f in enumerate(plot_files):
        row = (k // 2) * 28
        col = (k % 2) * 12
        ws.insert_image(row, col, str(f), {"x_scale": 0.9, "y_scale": 0.9})

    for name, frame in [("Patient_Records", df), ("Data_Dictionary", dictionary),
                        ("Genotype_Summary", genotype_summary), ("Numerical_Pairs", pairs_df)]:
        sheet = xw.sheets[name]
        sheet.freeze_panes(1, 0)
        for c, col in enumerate(frame.columns):
            width = max(len(str(col)), frame[col].astype(str).str.len().max()) + 2
            sheet.set_column(c, c, min(width, 60))

print(f"\nSaved {XLSX} and {len(plot_files)} plots in {PLOT_DIR}")
print(genotype_summary.to_string(index=False))
