import pandas as pd
import sys
sys.path.append("../utoronto_demo")
sys.path.append("..\\utoronto_demo\\status")

import pandas as pd

output_file = input_file = "./status/peroxide_assay.txt"

# Load the CSV into a DataFrame
df = pd.read_csv(input_file)

# Add the new columns by copying existing data
df["home_location"] = df["location"]
df["home_location_index"] = df["location_index"]

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Updated CSV saved as '{output_file}'")
