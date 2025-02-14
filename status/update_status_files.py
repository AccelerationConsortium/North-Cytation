import pandas as pd
import sys
sys.path.append("../utoronto_demo")
sys.path.append("..\\utoronto_demo\\status")

def process_vial_data(input_file, output_file):
    # Read the tab-separated file
    df = pd.read_csv(input_file, sep=',')
    
    # Add vial_type column based on location
    if 'location' in df.columns:
        df['vial_type'] = df['location'].map({'main_8mL_rack': '8_mL', 'large_vial_rack': '20_mL'})
    
    # Ensure correct column order
    expected_columns = ['vial_index','vial_name', 'location','location_index','vial_volume', 'capped', 'cap_type', 'vial_type']
    df = df[[col for col in expected_columns if col in df.columns]]
    
    # Save as comma-separated file
    df.to_csv(output_file, index=False)
    
    print(f"Processed file saved as {output_file}")


# Example usage
input_file = "../utoronto_demo/status/color_matching_vials.txt"  # Replace with the actual file path
output_file = "../utoronto_demo/status/color_matching_vials.txt"  # Replace with the desired output file name
process_vial_data(input_file, output_file)
