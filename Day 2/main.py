import numpy as np
import pandas as pd

df = pd.read_csv("inventory_processed.csv")

stock_series = df.set_index("ItemName")["Stock"]
print("--- Initial Stock Series (Custom Index) ---")
print(stock_series.head())

print("\n--- Inventory Dataset Overview ---")
print(df)
print("\nDimensions:", df.shape)
print("Columns:", list(df.columns))
print("\nData Types:")
print(df.dtypes)
print("\nDataFrame Summary:")
df.info()

print("\n--- ItemName Column ---")
print(df["ItemName"])

print("\n--- Multi-Column Projection (ItemCode, Category, Stock, UnitPrice) ---")
print(df[["ItemCode", "Category", "Stock", "UnitPrice"]])

print("\n--- Label-based Selection (loc: index 0) ---")
print(df.loc[0])

print("\n--- Label-based Range Slice (loc: 1 to 3) ---")
print(df.loc[1:3])

print("\n--- Position-based Selection (iloc: first row) ---")
print(df.iloc[0])

print("\n--- Position-based Slice (iloc: top 3 rows) ---")
print(df.iloc[:3])

print("\n--- Value Extraction (Row 2, UnitPrice) ---")
print(df.loc[2, "UnitPrice"])

print("\n--- Filter: Stock >= 30 ---")
print(df[df["Stock"] >= 30])

print("\n--- Filter: Category == 'Peripherals' ---")
print(df[df["Category"] == "Peripherals"])

print("\n--- Compound Filter: Category == 'Peripherals' & UnitPrice > 100 ---")
filtered_cohort = df[(df["Category"] == "Peripherals") & (df["UnitPrice"] > 100)]
print(filtered_cohort)

print("\n--- Row-wise Traversal ---")
for idx, entry in df.iterrows():
    print(f"[{entry['ItemCode']}] {entry['ItemName']} | Stock: {entry['Stock']} | Price: ${entry['UnitPrice']}")

dirty_df = df[["ItemCode", "ItemName", "Stock", "UnitPrice"]].copy()
dirty_df.loc[1, "Stock"] = np.nan
dirty_df.loc[3, "UnitPrice"] = np.nan
dirty_df.loc[4, "Stock"] = np.nan

print("\n--- Sample Missing Data Simulation ---")
print(dirty_df.head(6))

print("\n--- Null Value Matrix ---")
print(dirty_df.isnull().head(6))

print("\n--- Missing Count per Feature ---")
print(dirty_df.isnull().sum())

print("\n--- Records after Complete-Case Dropping (dropna) ---")
print(dirty_df.dropna().head(6))

imputed_df = dirty_df.copy()
imputed_df["Stock"] = imputed_df["Stock"].fillna(round(imputed_df["Stock"].mean()))
imputed_df["UnitPrice"] = imputed_df["UnitPrice"].fillna(round(imputed_df["UnitPrice"].mean(), 2))

print("\n--- Dataset post Imputation ---")
print(imputed_df.head(6))

df["RestockBonus"] = df["Stock"] + 5
print("\n--- Stock with Restock Bonus (+5) ---")
print(df[["ItemCode", "Stock", "RestockBonus"]].head())

df = df.rename(columns={"UnitPrice": "PriceUSD", "Category": "ProductType"})

print("\n--- Sorted Ascending by PriceUSD ---")
print(df.sort_values("PriceUSD").head())

print("\n--- Sorted Descending by PriceUSD ---")
print(df.sort_values("PriceUSD", ascending=False).head())

print(f"\nPrice Range: Low = ${df['PriceUSD'].min():.2f} | High = ${df['PriceUSD'].max():.2f}")

print("\nDistinct Product Types:")
print(df["ProductType"].unique())
print(f"Total Unique Product Types: {df['ProductType'].nunique()}")

df["PriceTier"] = df["PriceUSD"].apply(lambda p: "Premium" if p >= 500 else ("Mid-range" if p >= 150 else "Budget"))
df["DiscountedPrice"] = df["PriceUSD"].apply(lambda p: round(p * 0.90, 2))

print("\n--- Enhanced DataFrame with Price Tier & Discounted Price ---")
print(df[["ItemCode", "ItemName", "ProductType", "PriceUSD", "PriceTier", "DiscountedPrice"]].head())

print("\n--- Metric Aggregates for InventoryValue ---")
print(f"Total Inventory Valuation : ${df['InventoryValue'].sum():,.2f}")
print(f"Average Inventory Value   : ${df['InventoryValue'].mean():,.2f}")
print(f"Median Inventory Value    : ${df['InventoryValue'].median():,.2f}")
print(f"Standard Deviation        : ${df['InventoryValue'].std():,.2f}")
print(f"Total Products Count      : {df['InventoryValue'].count()}")

print("\n--- Statistical Overview ---")
print(df[["Stock", "PriceUSD", "InventoryValue"]].describe())

print("\n--- Mean Inventory Value by Product Type ---")
print(df.groupby("ProductType")["InventoryValue"].mean())

print("\n--- Min & Max Price by Product Type ---")
print(df.groupby("ProductType")["PriceUSD"].agg(["min", "max"]))

print("\n--- Comprehensive Category Breakdown ---")
category_analysis = df.groupby("ProductType").agg(
    Avg_Price=("PriceUSD", "mean"),
    Total_Stock=("Stock", "sum"),
    Total_Valuation=("InventoryValue", "sum"),
    Item_Count=("ItemCode", "count")
)
print(category_analysis)

warehouse_specs = df[["ItemCode", "ItemName", "Stock"]].head(5).copy()
supplier_info = pd.DataFrame({
    "ItemCode": ["SKU-1001", "SKU-1002", "SKU-1003", "SKU-1004", "SKU-1005"],
    "Supplier": ["Sony Electronics", "Apple Inc", "Samsung Direct", "Dell Distribution", "Logitech Global"],
    "LeadDays": [5, 3, 4, 7, 2]
})

merged_inventory = pd.merge(warehouse_specs, supplier_info, on="ItemCode")
print("\n--- Merged Warehouse & Supplier Records ---")
print(merged_inventory)

batch_a = df[["ItemCode", "ItemName", "PriceUSD"]].iloc[:3]
batch_b = df[["ItemCode", "ItemName", "PriceUSD"]].iloc[3:6]
stacked_batches = pd.concat([batch_a, batch_b], ignore_index=True)
print("\n--- Vertically Concatenated SKU Batches ---")
print(stacked_batches)

audit_metadata = pd.DataFrame({"AuditStatus": ["Verified", "Verified", "Pending", "Verified", "Verified", "Pending"]})
column_stacked = pd.concat([stacked_batches, audit_metadata], axis=1)
print("\n--- Horizontally Concatenated Inventory Audit ---")
print(column_stacked)

base_indexed = warehouse_specs.set_index("ItemCode")
supplier_indexed = supplier_info.set_index("ItemCode")
joined_dataset = base_indexed.join(supplier_indexed)
print("\n--- Index Joined Inventory Dataset ---")
print(joined_dataset)

df.to_csv("inventory_final_export.csv", index=False)
loaded_csv = pd.read_csv("inventory_final_export.csv")
print("\n--- Re-ingested from Final Export CSV ---")
print(loaded_csv.head(3))

try:
    df.to_excel("inventory_final_export.xlsx", index=False)
    loaded_excel = pd.read_excel("inventory_final_export.xlsx")
    print("\n--- Re-ingested from Final Export Excel ---")
    print(loaded_excel.head(3))
except ImportError:
    print("\n[Notice] openpyxl library not installed; skipping Excel export/import.")