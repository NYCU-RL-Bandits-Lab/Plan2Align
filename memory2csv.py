import os
import json
import pandas as pd
import argparse

def process_csv(num, input_csv, output_csv, column_name):
    # Read the input CSV file
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading {input_csv}: {e}")
        return

    # Add the target column
    df[column_name] = None

    # Process each row by using the index to construct the JSON file path
    for idx, row in df.iterrows():
        json_path = os.path.join(os.path.splitext(input_csv)[0], f"metadata_{idx}_iter_{num}.json")

        # Check if the file exists
        if not os.path.exists(json_path):
            print(f"File not found: {json_path}")
            df.at[idx, column_name] = None
            continue

        # Open and read the JSON file
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            df.at[idx, column_name] = None
            continue

        # Extract the value from final_translations_record
        final_record = data.get("final_translations_record", [])
        if isinstance(final_record, list) and len(final_record) > 0:
            value = final_record[0]
        else:
            value = None

        # Write the value into the target column
        df.at[idx, column_name] = value

    # Save the result to output CSV
    try:
        df.to_csv(output_csv, index=False)
        print(f"Saved successfully: {output_csv}")
    except Exception as e:
        print(f"Error saving {output_csv}: {e}")


# Example command: python memory2csv.py --num 5 --input_csv valid_en_ja.csv --output_csv eval_en_ja.csv --column_name mpc
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process CSV and extract data from JSON files.')
    parser.add_argument('--num', type=int, required=True, help='Iteration number used in JSON filenames')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--column_name', type=str, required=True, help='Column name to store extracted values')

    args = parser.parse_args()

    process_csv(args.num, args.input_csv, args.output_csv, args.column_name)