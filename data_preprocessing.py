import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.max_colwidth', None)  # show full column width

def create_dataset():
    columns = ["id", "audio_file", "text", "age", "sex", "mmse", "language", "image", "label"]
    df = pd.DataFrame(columns=columns)

    # Load existing data
    df_parquet = pd.read_parquet("dataset/taukadial/transcription/translation_train.parquet")
    df_csv = pd.read_csv("dataset/images/label.csv")

    images = ["", "image-1.jpg", "image-2.jpg", "image-3.jpg",
            "image-4.png", "image-5.png", "image-6.png"]

    for i in range(len(df_csv)):
        row_csv = df_csv.iloc[i]  # get i-th row
        
        id = row_csv["tkdname"].split(".")[0]
        
        # Get transcribed text from Parquet
        filtered = df_parquet.loc[df_parquet["file_name"] == id+".wav", "transcribed_text"]
        if not filtered.empty:
            text = filtered.iloc[0]
        else:
            text = ""  # or handle missing case

        # Map image index to filename
        img_idx = int(row_csv["picture"]) if pd.notna(row_csv["picture"]) else 0

        df.loc[len(df)] = [
            id,
            id+".wav",
            text,
            row_csv["age"],
            row_csv["sex"],
            row_csv["mmse"],
            row_csv["language"],
            images[img_idx],
            row_csv["dx"]
        ]
    return df
        
def plot_distribution(df):
    # Count the occurrences of each class
    class_counts = df['label'].value_counts()

    # Plot
    class_counts.plot(kind='bar')  # or 'pie', 'barh', etc.
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def main():
    #df = create_dataset()
    #print(len(df))
    df = pd.read_csv("dataset/combined_dataset.csv")

    plot_distribution(df)

    # Save as CSV
    df.to_csv("dataset/combined_dataset.csv", index=False)
