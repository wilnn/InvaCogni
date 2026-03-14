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
    df_parquet = pd.read_parquet("dataset/taukadial/taukadial/transcription/translation_train.parquet")
    df_csv = pd.read_csv("dataset/taukadial/images/label.csv")

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
    df.to_csv('dataset/taukadial/final_combined_dataset.csv', index=False)  # index=False prevents saving the row numbers

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
        
        plt.savefig(f"dataset/taukadial/{n}_distribution.png", dpi=300)
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
                
                plt.savefig(f"dataset/taukadial/{n}_{nn}_distribution.png", dpi=300)
                plt.show()
                plt.close()

        else:
            plt.hist(df[n.lower()], bins=15, edgecolor='black')
            plt.xlabel(f'{n}')
            plt.ylabel('Frequency')
            plt.title(f'{n} Distribution')

            plt.savefig(f"dataset/taukadial/{n}_distribution.png", dpi=300)
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

        plt.savefig(f"dataset/taukadial/{n}_NC_and_MCI_distribution.png", dpi=300)
        plt.show()
        plt.close()

def plot_pos_pie(pos_counts, lang, title="POS Distribution"):
    labels = pos_counts.keys()
    sizes = pos_counts.values()
    
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 8}, startangle=140)

    plt.title(lang.capitalize() + " "+ title)

    plt.savefig(f"dataset/taukadial/{lang}_pos_distribution.png", dpi=300)
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
    audio_folder = "dataset/taukadial/taukadial/train"
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
    df = pd.read_csv("dataset/taukadial/final_combined_dataset.csv")
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

#main()

def make_prepare_dataset():
    '''df = pd.read_csv("/home/public/htnguyen/projects/InvaCogni/dataset/prepare/acoustic/acoustic_data/train_labels.csv")
    print(len(df))
    print(df.columns)
    folder = "/home/public/htnguyen/projects/InvaCogni/dataset/prepare/acoustic/train_audios"
    num_files = sum(os.path.isfile(os.path.join(folder, f)) for f in os.listdir(folder))
    print(num_files)

    csv_path = "/home/public/htnguyen/projects/InvaCogni/dataset/prepare/acoustic/acoustic_data/train_labels.csv"
    mp3_folder = "/home/public/htnguyen/projects/InvaCogni/dataset/prepare/acoustic/train_audios"

    df = pd.read_csv(csv_path)

    missing = []
    bad = []
    ok = 0

    for uid in df["uid"].astype(str):
        mp3_path = os.path.join(mp3_folder, f"{uid}.mp3")

        # 1) check existence
        if not os.path.exists(mp3_path):
            missing.append(uid)
            continue

        # 2) check can open (basic check)
        try:
            with open(mp3_path, "rb") as f:
                f.read(16)  # read first few bytes
            ok += 1
        except Exception:
            bad.append(uid)

    print(f"Total uids: {len(df)}")
    print(f"OK mp3 files: {ok}")
    print(f"Missing mp3 files: {len(missing)}")
    print(f"Bad/unreadable mp3 files: {len(bad)}")

    parquet = "./dataset/prepare/acoustic/audio_to_text_w_language_train.parquet"
    parquet = pd.read_parquet(parquet)
    #print(len(parquet))
    #print(parquet.columns)
    #print(parquet['language'].iloc[0])'''
    

    '''
    id,audio_file,text,age,sex,mmse,language,image,label
    taukdial-002-1,taukdial-002-1.wav," Yes. Do you",72,F,29,english,image-1.jpg,NC
    
    parquet
    'file_name', 'transcribed_text', 'language'
    gyvr, This looks familiar. A woman drying dishes., en (cn)

    label
    uid            aaop
    diagnosis_control     0.0
    diagnosis_mci         1.0
    diagnosis_adrd        0.0
    Name: 0, dtype: object
    Index(['uid', 'diagnosis_control', 'diagnosis_mci', 'diagnosis_adrd'], dtype='object')
    
    
    metadata
    uid                                        aaop
    age                                          72
    gender                                   female
    split                                     train
    hash           e4ed5943c8460c2cf324d4a23c7d6fa1
    filesize_kb                             186.144
    Name: 0, dtype: object
    Index(['uid', 'age', 'gender', 'split', 'hash', 'filesize_kb'], dtype='object')
    '''

    '''
    lb['diagnosis'] = (
            lb['diagnosis_control'] * 0 +
            lb['diagnosis_mci'] * 1 +
            lb['diagnosis_adrd'] * 2
        ).astype(int)

    '''

    df_parquet = pd.read_parquet("./dataset/prepare/acoustic/audio_to_text_w_language_train.parquet")
    df_label = pd.read_csv("./dataset/prepare/acoustic/acoustic_data/train_labels.csv")
    df_metadata = pd.read_csv("./dataset/prepare/acoustic/acoustic_data/metadata.csv")
    
    df_parquet = df_parquet[df_parquet["language"] != "UNKNOWN"] # 1 example is unknown
    df_parquet.loc[df_parquet["language"] == "vi", "transcribed_text"] = "no a. no a mucho tiempo que vivía un hidalgo de los... de los de lanza en astillero, adarga adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero"
    df_parquet.loc[df_parquet["language"] == "vi", "language"] = "es"
    df_parquet.loc[df_parquet["language"] == "af", "transcribed_text"] = "Get a cookie and a cookie jar, right? Mothers sink overflowing while she's doing dishes and girl's waiting to have a cookie from the cookie jar while the boy is falling. Mother's wiping dishes, mothers standing in water, mothers looking out the window."
    df_parquet.loc[df_parquet["language"] == "af", "language"] = "en"
    df_parquet.loc[df_parquet["language"] == "pt", "transcribed_text"] = "Yo no me... no me de lo... de lo de... prueba... prueba... tiempo... tiempo que viví a una... luz de... m-m-m-m-m-m... me... los de la flo... la flo... esos son seis... que... que... que... que..."
    df_parquet.loc[df_parquet["language"] == "pt", "language"] = "es"
    df_parquet.loc[df_parquet["language"] == "ca", "language"] = "es"
    df_parquet = df_parquet[df_parquet["file_name"] != "xoct"]
    df_parquet.loc[df_parquet["language"] == "tl", "transcribed_text"] = "The stool is tilting as the kid is taking cookies to give to his sister. The water is running over the sink. The mother has to be blind. She dropped the dishes. What else am I supposed to find out? What else am I supposed to know? All happening? Like I said, the water running over on the floor. Mother standing in it. She's washing dishes.  She'd have to be blind not to know what's going on in that house."
    df_parquet.loc[df_parquet["language"] == "tl", "language"] = "en"
    df_parquet.loc[df_parquet["file_name"] == "ufed", "transcribed_text"] = "This is how she will find her. Mother's washing, drying the dishes. And Stu's upset. You know, she's getting all given. Cup and saucer sitting there in a plate ready to eat. Cookie jar is right here. There's a pineapple over there, spilled it, knocks it down. I don't know what that is. Looks like a mouse."
    df_parquet.loc[df_parquet["file_name"] == "ufed", "language"] = "en"
    df_parquet.loc[df_parquet["file_name"] == "vbxv", "transcribed_text"] = "The little boy is up on a stool reaching for a cookie, and the stool is ready to fall. He has one hand in the cookie jar and a cookie in his left hand. A girl is standing beside him, reaching for the cookie in his hand with her right finger held up to her lips. There is also a woman, the mother, standing at the sink looking out the window. The water is running over the top of the sink and onto the floor while she is looking away."
    df_parquet.loc[df_parquet["file_name"] == "vbxv", "language"] = "en"
    df_parquet.loc[df_parquet["file_name"] == "wbzl", "transcribed_text"] = "This is fine on the left. Okay, and the kitchen's draining on the left, is it? Yes. And Jimmy is reaching for the cookie—cookie jar. Uh-huh. Getting the cookie jar. Pardon? Cookie jar. Okay. She's doing the dishes, the boy might fall off of that stool he's on, okay. And the water is splashing out all over the floor and she doesn't know it. Uh-huh. And the boy might look up for the cookie jar and he might bang his head on the"
    df_parquet.loc[df_parquet["file_name"] == "wbzl", "language"] = "en"

    
    #print(df_parquet.iloc)
    #print(df_parquet.columns)
    #print(df_label.iloc[0])
    #print(df_label.columns)
    #print(df_metadata.iloc[0])
    #print(df_metadata.columns)
    columns = ['id','audio_file','text','age','sex','language','label']
    rows = []

    for uid in df_label["uid"]:
        subset = df_parquet.loc[df_parquet["file_name"] == uid, "transcribed_text"]
        if not subset.empty:
            text = subset.iloc[0].strip()  # safe: get the value
        else:
            continue
        age = df_metadata.loc[(df_metadata["uid"] == uid) & (df_metadata["split"] == "train"), "age"].iloc[0]
        sex = df_metadata.loc[(df_metadata["uid"] == uid) & (df_metadata["split"] == "train"), "gender"].iloc[0]
        sex = "F" if sex == "female" else "M"

        language = df_parquet.loc[df_parquet["file_name"] == uid, "language"].iloc[0]
        language = "english" if language == "en" else "non_english"
        row = df_label.loc[df_label["uid"] == uid].iloc[0]  # get the single row as Series

        label = (
            row['diagnosis_control'] * 0 +
            row['diagnosis_mci'] * 1 +
            row['diagnosis_adrd'] * 2
        ).astype(int)
        if label == 0:
            label = "NC" 
        elif label == 1:
            label = "MCI"
        else:
            label = "ADRD" # ADRD = Alzheimer’s Disease and Related Dementias

        rows.append((uid, f"{uid}.mp3", text, age, sex, language, label))

    df = pd.DataFrame(rows, columns=columns)
    #print(df['language'].value_counts())
    #df.to_csv("./dataset/prepare/final_combined_dataset.csv", index=False)
    #print(f"label {df["label"].value_counts()}")
    values = df.loc[df["language"] == "english", "id"].tolist()
    avg_age = df_metadata.loc[df_metadata["uid"].isin(values), "age"].mean()
    print(f"english avg age: {avg_age}")
    values = df.loc[df["language"] == "non_english", "id"].tolist()
    avg_age = df_metadata.loc[df_metadata["uid"].isin(values), "age"].mean()
    print(f"non english avg age: {avg_age}")

    print(f"male english count: {df[(df["sex"] == "M") &(df["language"] == "english")].shape[0]}")
    print(f"female english count: {df[(df["sex"] == "F") &(df["language"] == "english")].shape[0]}")
    print(f"male non english count: {df[(df["sex"] == "M") &(df["language"] == "non_english")].shape[0]}")
    print(f"female non english count: {df[(df["sex"] == "F") &(df["language"] == "non_english")].shape[0]}")   

    print(f"MCI english count: {df[(df["label"] == "MCI") &(df["language"] == "english")].shape[0]}")
    print(f"NC english count: {df[(df["label"] == "NC") &(df["language"] == "english")].shape[0]}")
    print(f"ADRD english count: {df[(df["label"] == "ADRD") &(df["language"] == "english")].shape[0]}")

    print(f"MCI non_english count: {df[(df["label"] == "MCI") &(df["language"] == "non_english")].shape[0]}")
    print(f"NC non_english count: {df[(df["label"] == "NC") &(df["language"] == "non_english")].shape[0]}")
    print(f"ADRD non_english count: {df[(df["label"] == "ADRD") &(df["language"] == "non_english")].shape[0]}")
    print(len(df))
    print(df_metadata['age'].median())

# ----------------------------
# Load dataset
# ----------------------------
# Replace with your real file name/path
df = pd.read_csv("/home/public/htnguyen/projects/InvaCogni/dataset/prepare/final_combined_dataset.csv")

# ----------------------------
# Basic Data Cleaning (Optional but recommended)
# ----------------------------
# Remove rows with missing values in key columns
df = df.dropna(subset=["age", "sex", "language", "label"])

# ----------------------------
# 1. Average Age per Label
# ----------------------------
avg_age = df.groupby("label")["age"].mean()
print("\n===== Average Age per Label =====")
print(avg_age)

# ----------------------------
# 2. Number of People per Sex within Each Label
# ----------------------------
sex_counts = df.groupby(["label", "sex"]).size().unstack(fill_value=0)

print("\n===== Sex Counts per Label =====")
print(sex_counts)

# ----------------------------
# 3. Number of People per Language within Each Label
# ----------------------------
language_counts = df.groupby(["label", "language"]).size().unstack(fill_value=0)

print("\n===== Language Counts per Label =====")
print(language_counts)

print(df['age'].mean())

print("\nResults saved as CSV files!")