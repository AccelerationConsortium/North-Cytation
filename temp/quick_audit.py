import pandas as pd

df200 = pd.read_csv(r'C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\200uL\incremental_results.csv')
df1000 = pd.read_csv(r'C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\1000uL\incremental_results.csv')

# Q1
for name, df in [("200uL", df200), ("1000uL", df1000)]:
    valid = df[df["run_type"].notna()]
    covered = set(valid["row_index"].dropna().astype(int).unique())
    missing = set(range(5120)) - covered
    print("COVERAGE", name, str(len(covered)) + "/5120", "missing:", sorted(missing))
    print("NaN rows:", df["run_type"].isna().sum())

# Q3
r200 = [4538, 4539, 4540, 4541, 4542]
r1000 = ([101, 105, 129, 218, 235, 251, 312, 324, 342, 385, 392, 400, 423, 424, 456,
          485, 497, 502, 594, 633, 714, 722, 732, 744, 775, 843, 853, 857,
          1019, 1190, 1552, 1674, 1731, 1855, 2041,
          3175, 3836, 3884, 4028, 4205, 4649, 4963]
         + list(range(4314, 4360)) + list(range(4462, 4482)))
bad200 = [i for i in r200 if "redo" not in df200[df200["row_index"] == i]["run_type"].tolist()]
bad1000 = [i for i in r1000 if "redo" not in df1000[df1000["row_index"] == i]["run_type"].tolist()]
print("BAD 200uL:", bad200)
print("BAD 1000uL:", bad1000)

# Q2 - delete NaN rows from 200uL
path200 = r'C:\Users\Imaging Controller\Desktop\utoronto_demo\output\glycerol_sobol_campaign\200uL\incremental_results.csv'
before = len(df200)
df200_clean = df200[df200["run_type"].notna()]
df200_clean.to_csv(path200, index=False)
print("Removed NaN rows from 200uL:", before - len(df200_clean), "-> now", len(df200_clean), "rows")
