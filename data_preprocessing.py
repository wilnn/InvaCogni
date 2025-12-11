import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import string
import numpy as np
#import hanlp
import jieba
import stanza
from collections import Counter
import librosa
import os
#nltk.download('punkt_tab')
# English model
#stanza.download('en')

# Chinese model
#stanza.download('zh')

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
    df.to_csv('dataset/final_combined_dataset.csv', index=False)  # index=False prevents saving the row numbers

    return df
        
def plot_distribution(df):
    l = ["Label", "Gender", "Language"]
    for n in l:
        # Count the occurrences of each class
        k = n.lower() if n != "Gender" else "sex" 
        class_counts = df[k].value_counts()
        summ = class_counts.sum()
        counts = list(class_counts.items())
        print(f"\033[32m{counts}\033[0m")
        for nn in counts:
            print(f"\033[32m{nn[0]}: {nn[1]/summ*100:.1f}%\033[0m")

        # Plot
        class_counts.plot(kind='bar')  # or 'pie', 'barh', etc.
        plt.title(f"{n} Distribution")
        plt.xlabel(n)
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        
        plt.savefig(f"dataset/{n}_distribution.png", dpi=300)
        plt.show()
        plt.close()
    
    l = ["Age", "MMSE"]
    for n in l:
        if n == "MMSE":
            for nn in df['language'].unique().tolist():
                filtered_df = df[df['language'] == nn]
                
                plt.hist(filtered_df[n.lower()], bins=10, edgecolor='black')
                plt.xlabel(f'{n}')
                plt.ylabel('Frequency')
                plt.title(f'{n} Distribution for {nn.capitalize()}')
                
                plt.savefig(f"dataset/{n}_{nn}_distribution.png", dpi=300)
                plt.show()
                plt.close()

        else:
            plt.hist(df[n.lower()], bins=15, edgecolor='black')
            plt.xlabel(f'{n}')
            plt.ylabel('Frequency')
            plt.title(f'{n} Distribution')

            plt.savefig(f"dataset/{n}_distribution.png", dpi=300)
            plt.show()
            plt.close()

def plot_distribution_against_label(df):
    #print("88888888888888888888")
    l = ["Gender", "Language"]
    for n in l:
        dd = n if n != "Gender" else "sex"
        counts = df.groupby([dd.lower(), 'label']).size().unstack(fill_value=0)
        #print(counts)
        # Rename columns for clarity (optional)
        #counts.columns = ['NC', 'MCI']
        #print(counts.sum())
        co = counts.div(counts.sum(axis=1), axis=0)
        #co = co.values
        #print(co)
        #exit(0)
        for id, ii in enumerate(co.index.tolist()):
            print(f"\033[32mpercent {ii} that has MCI: {co.values[id][0]*100:.1f}%\033[0m")
        #exit(0)
        # Plot grouped bar chart
        counts.plot(kind='bar', figsize=(8,6))
        plt.title(f"Number of NC and MCI by {n}")
        plt.xlabel(n)
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.legend(title="Label")

        plt.savefig(f"dataset/{n}_NC_and_MCI_distribution.png", dpi=300)
        plt.show()
        plt.close()

def plot_pos_pie(pos_counts, lang, title="POS Distribution"):
    labels = pos_counts.keys()
    sizes = pos_counts.values()
    
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 8}, startangle=140)

    plt.title(lang.capitalize() + " "+ title)

    plt.savefig(f"dataset/{lang}_pos_distribution.png", dpi=300)
    plt.show()
    plt.close()


def analyze_text(df):

    lang = ["english", "chinese"]
    
    for n in lang:
        d = {}
        pos_counts = Counter()

        print(f"\033[32m{n}:\033[0m")
        filtered_df = df[df['language'] == n]
        # Get the values of 'text' column as a list
        text_list = filtered_df['text'].tolist()
        word_counts = 0

        if n == "english":
            nlp_en = stanza.Pipeline('en', processors='tokenize,pos')
            for t in text_list:
                tokens = word_tokenize(t)
                for tt in tokens:
                    token = tt.strip().lower()
                    if token not in string.punctuation:
                        word_counts +=1
                        if token not in d:
                            d[token] = 1
                        else:
                            d[token] +=1
                
                doc = nlp_en(t)
                for sent in doc.sentences:
                    for word in sent.words:
                        pos_counts[word.upos] += 1  # UPOS is the universal POS tag
            
            plot_pos_pie(pos_counts, n)

        if n == "chinese":
            #tokenizer = hanlp.load('CTB6_CONVSEG')
            nlp_zh = stanza.Pipeline('zh', processors='tokenize,pos')
            for t in text_list:
                tokens = jieba.cut(t)
                for tt in tokens:
                    token = tt.strip().lower()
                    if token not in string.punctuation:
                        word_counts +=1
                        if token not in d:
                            d[token] = 1
                        else:
                            d[token] +=1
                doc = nlp_en(t)
                for sent in doc.sentences:
                    for word in sent.words:
                        pos_counts[word.upos] += 1  # UPOS is the universal POS tag
            
            plot_pos_pie(pos_counts, n)

        #print(d)
        print(f"\033[32mNumber of unique words: {len(d)}\033[0m")
        print(f"\033[32mAverage number of words per sample: {word_counts//len(text_list)}\033[0m")
        print("\033[32m-------------------------------\033[0m")
        #ll = [[k, v] for k, v in d.items()]
        #ldx = [l[1] for l in ll]
        #sorted_indices = np.argsort(ldx)


def analyze_audio():
    audio_folder = "dataset/taukadial/train"
    lengths = []
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(".wav"):  # adjust extension if needed
            file_path = os.path.join(audio_folder, file_name)
            y, sr = librosa.load(file_path, sr=None)  # load audio
            duration = librosa.get_duration(y=y, sr=sr)
            lengths.append(duration)

    average_length = np.mean(lengths)
    print(f"\033[32mAverage audio length: {average_length:.2f} seconds\033[0m")

def main():
    df = create_dataset()
    #print(len(df))
    df = pd.read_csv("dataset/final_combined_dataset.csv")
    print(f"\033[32mAverage age: {df['age'].mean()}\033[0m")
    print(f"\033[32mAge range from {df['age'].min()} to {df['age'].max()}\033[0m")
    print(f"\033[32mAverage MMSE: {df['mmse'].mean()}\033[0m")
    print(f"\033[32mMMSE range from {df['mmse'].min()} to {df['mmse'].max()}\033[0m")
    plot_distribution(df)
    plot_distribution_against_label(df)
    print("\033[32m##################\033[0m")
    analyze_text(df)
    # Save as CSV
    #df.to_csv("dataset/final_combined_dataset.csv", index=False)
    print("\033[32mAverage image size: 656x516\033[0m")
    print("\033[32m##################\033[0m")
    analyze_audio()

main()