import pandas as pd

def clean_csv(file_path, output_path):
    # Load the CSV file without headers
    df = pd.read_csv(file_path, header=None)
    print("Initial data load:")
    print(df.head())  # Inspect the first few rows to understand the structure

    # Specifying column names based on your observed data structure
    column_names = ['Frame', 'Path', 'Extra', 'Label', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax']
    df.columns = column_names[:len(df.columns)]  # Adjust column names to match data

    # Drop any rows where critical data might be missing (e.g., no bounding box coordinates)
    df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax'], inplace=True)

    # Correct data shifting issues by checking data types or misplacements
    if df['xmin'].dtype == object:
        # Assume 'xmin' should be float, and non-float entries indicate row misalignment
        df = df[df['xmin'].apply(lambda x: x.replace('.', '', 1).isdigit())]

    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    df[['xmin', 'ymin', 'xmax', 'ymax']] = df[['xmin', 'ymin', 'xmax', 'ymax']].apply(pd.to_numeric, errors='coerce')

    print("Cleaned DataFrame head:")
    print(df.head())

    # Save the cleaned data to a new CSV file
    df.to_csv(output_path, index=False)

# Usage: (path to csv you want to be organized, new path to the new csv name and where you want it to be placed)
clean_csv('/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/yolov5/runs/detect/Angle3_Output/predictions.csv', 
          '/Users/ethanhaque/repos/Test Repo/Aviary-Bird-Project/processed_data/csv/cleaned_output3.csv')
