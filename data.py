import json
import pandas as pd
from pathlib import Path


def load_data(path: Path | str, verbose=False):
    """
    Load and flatten JSON data from the specified path into a Pandas DataFrame.
    
    Parameters
    ----------
    path : Path
        Path to the JSON file.
    verbose : bool, optional
        If True, prints metadata and status messages. Default is False.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        session_metadata = data.get('metadata', {})
        if verbose:
            print("✅ JSON file loaded successfully.")
            print("\n--- Session Metadata ---")
            for key, value in session_metadata.items():
                print(f"{key.ljust(20)}: {value}")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at path: {path}")
        print("Please check the file path and ensure Google Drive is correctly mounted.")
    except json.JSONDecodeError:
        print("❌ ERROR: Could not decode JSON. Ensure the file is valid.")

    # Faltten JSON into DataFrame
    flat_data = []
    session_name = session_metadata.get('sessionName', 'Unknown Session')

    # Iterate through each sample (color card)
    for sample in data.get('data', []):
        sample_number = sample.get('sampleNumber')

        # Iterate through each measurement (1 to 10) within that sample
        for capture_index, measurement in enumerate(sample.get('measurements', [])):

            # Create a dictionary for the current row
            row = {
                'session_name': session_name,
                'sample_number': sample_number,
                'capture_index': capture_index, # 0 to 9
                'lighting_condition': session_metadata.get('lightingCondition'),
                'reflective_surface': session_metadata.get('useReflectiveSurface'),

                # Sensor Data
                'pitch': measurement['angles']['pitch'],
                'roll': measurement['angles']['roll'],
            }

            # Extract Color Data (White and Color reticles, three radii each)

            # White Reticle Captures
            for radius in [0, 2, 4]:
                capture_key = f'r{radius}'
                color_data = measurement['white'].get(capture_key, {'r': 0, 'g': 0, 'b': 0})
                row[f'white_r{radius}_R'] = color_data['r']
                row[f'white_r{radius}_G'] = color_data['g']
                row[f'white_r{radius}_B'] = color_data['b']

            # Color Reticle Captures
            for radius in [0, 2, 4]:
                capture_key = f'r{radius}'
                color_data = measurement['color'].get(capture_key, {'r': 0, 'g': 0, 'b': 0})
                row[f'color_r{radius}_R'] = color_data['r']
                row[f'color_r{radius}_G'] = color_data['g']
                row[f'color_r{radius}_B'] = color_data['b']

            flat_data.append(row)

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(flat_data)
    if verbose:
        print(f"\n✅ Data flattened into DataFrame with {len(df)} rows (Total captures: 24 samples * 10 captures = 240 rows).")

    # Combine ground-truths (provided by the color cards)
    ground_truth_data = [
        {'sample_number': 1,  'label': 'Dark Skin',      'gt__R': 115, 'gt__G': 82,  'gt__B': 69},
        {'sample_number': 2,  'label': 'Light Skin',     'gt__R': 204, 'gt__G': 161, 'gt__B': 141},
        {'sample_number': 3,  'label': 'Blue Sky',       'gt__R': 101, 'gt__G': 134, 'gt__B': 179},
        {'sample_number': 4,  'label': 'Foliage',        'gt__R': 89,  'gt__G': 109, 'gt__B': 61},
        {'sample_number': 5,  'label': 'Blue Flower',    'gt__R': 141, 'gt__G': 137, 'gt__B': 194},
        {'sample_number': 6,  'label': 'Bluish Green',   'gt__R': 132, 'gt__G': 228, 'gt__B': 208},
        {'sample_number': 7,  'label': 'Orange',         'gt__R': 249, 'gt__G': 118, 'gt__B': 35},
        {'sample_number': 8,  'label': 'Purplish Blue',  'gt__R': 80,  'gt__G': 91,  'gt__B': 182},
        {'sample_number': 9,  'label': 'Moderate Red',   'gt__R': 222, 'gt__G': 91,  'gt__B': 125},
        {'sample_number': 10, 'label': 'Purple',         'gt__R': 91,  'gt__G': 63,  'gt__B': 123},
        {'sample_number': 11, 'label': 'Yellow Green',   'gt__R': 173, 'gt__G': 232, 'gt__B': 91},
        {'sample_number': 12, 'label': 'Orange Yellow',  'gt__R': 255, 'gt__G': 164, 'gt__B': 26},
        {'sample_number': 13, 'label': 'Blue',           'gt__R': 44,  'gt__G': 56,  'gt__B': 142},
        {'sample_number': 14, 'label': 'Green',          'gt__R': 74,  'gt__G': 148, 'gt__B': 81},
        {'sample_number': 15, 'label': 'Red',            'gt__R': 179, 'gt__G': 42,  'gt__B': 50},
        {'sample_number': 16, 'label': 'Yellow',         'gt__R': 250, 'gt__G': 226, 'gt__B': 21},
        {'sample_number': 17, 'label': 'Magenta',        'gt__R': 191, 'gt__G': 81,  'gt__B': 160},
        {'sample_number': 18, 'label': 'Cyan',           'gt__R': 6,   'gt__G': 142, 'gt__B': 172},
        {'sample_number': 19, 'label': 'White',          'gt__R': 252, 'gt__G': 252, 'gt__B': 252},
        {'sample_number': 20, 'label': 'Neutral 8',      'gt__R': 230, 'gt__G': 230, 'gt__B': 230},
        {'sample_number': 21, 'label': 'Neutral 6.5',    'gt__R': 200, 'gt__G': 200, 'gt__B': 200},
        {'sample_number': 22, 'label': 'Neutral 5',      'gt__R': 143, 'gt__G': 143, 'gt__B': 142},
        {'sample_number': 23, 'label': 'Neutral 3.5',    'gt__R': 100, 'gt__G': 100, 'gt__B': 100},
        {'sample_number': 24, 'label': 'Black',          'gt__R': 50,  'gt__G': 50,  'gt__B': 50},
    ]
    df_gt = pd.DataFrame(ground_truth_data)
    # Append the true RGB values to every row
    df = pd.merge(df, df_gt, on='sample_number', how='outer')

    if verbose:
        print("\n--- DataFrame Head (First 5 Rows) ---")
        print(df.head())
        print("\n--- DataFrame Information ---")
        print(df.info())
        # This checks the pitch/roll stability across all 240 measurements.
        print("\n--- Sensor Angle Statistics ---")
        print(df[['pitch', 'roll']].describe())

    return df


def displayDataFrameInfo(df):
    print("\n--- DataFrame Head (First 5 Rows) ---")
    print(df.head())

    print("\n--- DataFrame Information ---")
    print(df.info())

    # ### 3.1 Check Sensor Variability

    # This checks the pitch/roll stability across all 240 measurements.
    print("\n--- Sensor Angle Statistics ---")
    print(df[['pitch', 'roll']].describe())


# TODO: Remove this before submission.
if __name__ == "__main__":
    path = "Data/Baisu1.json"
    df = load_data(Path(path), verbose=False)
    print(df.head())
    print(df.info())
