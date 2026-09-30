import numpy as np
import pandas as pd

scores = pd.Series([74, 91, 83, 67, 95])
print("--- Initial Series ---")
print(scores)

student_ids = ["ST_101", "ST_102", "ST_103", "ST_104", "ST_105"]
student_names = ["Rohan", "Meera", "Vikram", "Ananya", "Karan"]
scores = pd.Series([74, 91, 83, 67, 95], index=student_names)
print("\n--- Series (Custom Index) ---")
print(scores)

records = {
    "StudentID": student_ids,
    "Name": student_names,
    "Age": [21, 20, 22, 19, 21],
    "Score": [74, 91, 83, 67, 95],
    "Stream": ["DataScience", "CyberSec", "DataScience", "Cloud", "CyberSec"],
    "City": ["Bangalore", "Delhi", "Mumbai", "Pune", "Chennai"]
}

df = pd.DataFrame(records)
print("\n--- Student DataFrame ---")
print(df)
print("\nDimensions:", df.shape)
print("Columns:", list(df.columns))
print("\nData Types:")
print(df.dtypes)
print("\nDataFrame Summary:")
df.info()

print("\n--- Name Column ---")
print(df["Name"])

print("\n--- Multi-Column Projection (Name, Stream, Score) ---")
print(df[["Name", "Stream", "Score"]])

print("\n--- Label-based Selection (loc: index 0) ---")
print(df.loc[0])

print("\n--- Label-based Range Slice (loc: 1 to 3) ---")
print(df.loc[1:3])

print("\n--- Position-based Selection (iloc: first row) ---")
print(df.iloc[0])

print("\n--- Position-based Slice (iloc: top 3 rows) ---")
print(df.iloc[:3])

print("\n--- Value Extraction (Row 2, Score) ---")
print(df.loc[2, "Score"])

print("\n--- Filter: Score >= 80 ---")
print(df[df["Score"] >= 80])

print("\n--- Filter: DataScience Stream ---")
print(df[df["Stream"] == "DataScience"])

print("\n--- Compound Filter: DataScience with Score > 75 ---")
filtered_cohort = df[(df["Stream"] == "DataScience") & (df["Score"] > 75)]
print(filtered_cohort)

print("\n--- Row-wise Traversal ---")
for idx, entry in df.iterrows():
    print(f"[{entry['StudentID']}] {entry['Name']} ({entry['Stream']}): {entry['Score']}")

raw_dirty_data = {
    "Name": ["Rohan", "Meera", "Vikram", "Ananya", "Karan"],
    "Age": [21, np.nan, 22, 19, np.nan],
    "Score": [74, 91, np.nan, 67, 95],
    "Attendance": [88.5, np.nan, 72.0, 95.0, 81.0]
}

dirty_df = pd.DataFrame(raw_dirty_data)
print("\n--- Raw Data with Missing Values ---")
print(dirty_df)

print("\n--- Null Value Matrix ---")
print(dirty_df.isnull())

print("\n--- Missing Count per Feature ---")
print(dirty_df.isnull().sum())

print("\n--- Records after Complete-Case Dropping (dropna) ---")
print(dirty_df.dropna())

imputed_df = dirty_df.copy()
numeric_cols = ["Age", "Score", "Attendance"]
for col in numeric_cols:
    imputed_df[col] = imputed_df[col].fillna(round(imputed_df[col].mean(), 1))

print("\n--- Dataset post Imputation ---")
print(imputed_df)

df["Score"] = df["Score"] + 3
print("\n--- Adjusted Scores (+3 Grace) ---")
print(df[["Name", "Score"]])

df = df.rename(columns={"Score": "FinalScore", "Stream": "Specialization"})

print("\n--- Sorted Ascending by FinalScore ---")
print(df.sort_values("FinalScore"))

print("\n--- Sorted Descending by FinalScore ---")
print(df.sort_values("FinalScore", ascending=False))

print(f"\nScore Range: Low = {df['FinalScore'].min()} | High = {df['FinalScore'].max()}")

print("\nDistinct Specializations:")
print(df["Specialization"].unique())
print(f"Total Specializations: {df['Specialization'].nunique()}")

df["Grade"] = df["FinalScore"].apply(lambda s: "A" if s >= 90 else ("B" if s >= 75 else "C"))
df["Percentage"] = df["FinalScore"].apply(lambda s: round((s / 100) * 100, 2))

print("\n--- Enhanced DataFrame with Grade & Percentage ---")
print(df)

print("\n--- Metric Aggregates ---")
print(f"Total Score Sum    : {df['FinalScore'].sum()}")
print(f"Average Score      : {df['FinalScore'].mean():.2f}")
print(f"Median Score       : {df['FinalScore'].median():.2f}")
print(f"Standard Deviation : {df['FinalScore'].std():.2f}")
print(f"Count              : {df['FinalScore'].count()}")

print("\n--- Statistical Overview ---")
print(df["FinalScore"].describe())

print("\n--- Mean FinalScore by Specialization ---")
print(df.groupby("Specialization")["FinalScore"].mean())

print("\n--- Min & Max Score by Specialization ---")
print(df.groupby("Specialization")["FinalScore"].agg(["min", "max"]))

print("\n--- Comprehensive Cohort Analysis ---")
cohort_analysis = df.groupby("Specialization").agg(
    Avg_Score=("FinalScore", "mean"),
    Max_Score=("FinalScore", "max"),
    Min_Score=("FinalScore", "min"),
    Total_Students=("FinalScore", "count")
)
print(cohort_analysis)

student_base = pd.DataFrame({
    "StudentID": ["ST_101", "ST_102", "ST_103"],
    "Name": ["Rohan", "Meera", "Vikram"]
})

academic_records = pd.DataFrame({
    "StudentID": ["ST_101", "ST_102", "ST_103"],
    "GPA": [3.8, 3.9, 3.4],
    "Credits": [120, 115, 118]
})

merged_records = pd.merge(student_base, academic_records, on="StudentID")
print("\n--- Merged Student Profiles ---")
print(merged_records)

batch_alpha = pd.DataFrame({"Name": ["Rohan", "Meera"], "GPA": [3.8, 3.9]})
batch_beta = pd.DataFrame({"Name": ["Vikram", "Ananya"], "GPA": [3.4, 3.7]})
stacked_batches = pd.concat([batch_alpha, batch_beta], ignore_index=True)
print("\n--- Vertically Concatenated Batches ---")
print(stacked_batches)

extra_metadata = pd.DataFrame({"Status": ["Active", "Active", "Probation", "Active"]})
column_stacked = pd.concat([stacked_batches, extra_metadata], axis=1)
print("\n--- Horizontally Concatenated Attributes ---")
print(column_stacked)

base_indexed = student_base.set_index("StudentID")
records_indexed = academic_records.set_index("StudentID")
joined_dataset = base_indexed.join(records_indexed)
print("\n--- Index Joined Dataset ---")
print(joined_dataset)

inventory = pd.read_csv("inventory_data.csv")
print("\n--- Ingested Kaggle Inventory Dataset ---")
print(inventory)

inventory["InventoryValue"] = (inventory["Stock"] * inventory["UnitPrice"]).round(2)
print("\n--- Inventory Summary with Calculated Valuation ---")
print(inventory.head())

inventory.to_csv("inventory_processed.csv", index=False)
loaded_csv = pd.read_csv("inventory_processed.csv")
print("\n--- Re-ingested from Processed CSV ---")
print(loaded_csv.head(3))

try:
    inventory.to_excel("inventory_data.xlsx", index=False)
    loaded_excel = pd.read_excel("inventory_data.xlsx")
    print("\n--- Re-ingested from Excel ---")
    print(loaded_excel.head(3))
except ImportError:
    print("\n[Notice] openpyxl library not installed; skipping Excel export/import.")