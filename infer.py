from ollama import chat
from ollama import ChatResponse
import pandas as pd
import progressbar
import os
import jiwer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to classify text difficulty
def classify_text_difficulty(text: str, model_name: str) -> str:
    response: ChatResponse = chat(model=model_name, messages=[
        {
            'role': 'system',
            'content': (
                'You are an expert in text difficulty classification. '
                'Classify the given text into one of the following levels of difficulty: '
                'Very Easy, Easy, Accessible, or Complex. '
                'Respond with only the difficulty level.'
            ),
        },
        {
            'role': 'user',
            'content': text,
        },
    ])
    return response['message']['content']


def load_dataset(path="../../data/Qualtrics_Annotations_formatB.csv"):
    df = pd.read_csv(path)
    return df

def infer_classification(dataset, model_name, output_path):
    bar = progressbar.ProgressBar(maxval=len(dataset))
    i = 0
    for index, row in dataset.iterrows():
        dataset.at[index, "difficulty"] = classify_text_difficulty(row["text"])
        i += 1
        bar.update(i)
    bar.finish()
    # save in csv format
    dataset.to_csv(output_path, index=False)
    return dataset

def save_confusion_matrix(y_true, y_pred, confusion_matrix_path): # output_path not used
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Very Easy", "Easy", "Accessible", "Complex"],
                yticklabels=["Très Facile", "Facile", "Accessible", "+Complexe"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.title("Matrice de Confusion")
    plt.show()

    # save
    plt.savefig(confusion_matrix_path)


def evaluate_classification(dataset, confusion_matrix_path):
    # Correction des valeurs erronées dans la colonne "difficulty"
    for index, row in dataset.iterrows():
        if row["difficulty"] not in ["Very Easy", "Easy", "Accessible", "Complex"]:
            # Calcul du CER pour chaque valeur candidate et sélection de la meilleure
            candidates = ["Very Easy", "Easy", "Accessible", "Complex"]
            cer_scores = [jiwer.cer(row["difficulty"][:max(len(row["difficulty"]), 30)], candidate) for candidate in candidates]
            dataset.at[index, "difficulty"] = candidates[cer_scores.index(min(cer_scores))]

    # Conversion des valeurs textuelles en numériques
    mapping_pred = {"Very Easy": 0, "Easy": 1, "Accessible": 2, "Complex": 3}
    mapping_gold = {"Très Facile": 0, "Facile": 1, "Accessible": 2, "+Complexe": 3}
    dataset["difficulty"] = dataset["difficulty"].map(mapping_pred)
    dataset["gold_score_20_label"] = dataset["gold_score_20_label"].map(mapping_gold)

    # Extraction des valeurs réelles et prédites
    y_pred = dataset["difficulty"]
    y_true = dataset["gold_score_20_label"]

    # Calcul des métriques globales
    global_accuracy = accuracy_score(y_true, y_pred)
    global_adjacent_accuracy = (abs(y_true - y_pred) <= 1).mean()
    global_macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Global Accuracy: {global_accuracy}")
    print(f"Global Adjacent Accuracy: {global_adjacent_accuracy}")
    print(f"Global Macro F1: {global_macro_f1}")

    # Calcul des métriques par classe (F1 classique pour chaque classe)
    for difficulty in [0, 1, 2, 3]:
        # Sélection des exemples dont la vérité terrain est la classe 'difficulty'
        idx = (y_true == difficulty)
        if idx.sum() == 0:
            continue

        # Accuracy locale (sur les exemples de la classe)
        class_accuracy = (y_pred[idx] == y_true[idx]).mean()

        # Adjacent accuracy locale (si la différence absolue <= 1)
        class_adjacent_accuracy = (abs(y_pred[idx] - y_true[idx]) <= 1).mean()

        # Calcul du F1 pour la classe en mode binaire (classe vs reste)
        y_true_binary = (y_true == difficulty).astype(int)
        y_pred_binary = (y_pred == difficulty).astype(int)
        class_f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)

        print()
        print(f"Difficulty: {difficulty}")
        print(f"  Accuracy: {class_accuracy}")
        print(f"  Adjacent Accuracy: {class_adjacent_accuracy}")
        print(f"  F1: {class_f1}")

    save_confusion_matrix(y_true, y_pred, confusion_matrix_path)

def get_difficulty_level(dataset_path, model_name, output_path):
    if os.path.exists(output_path):
        dataset = pd.read_csv(output_path)
    else:
        dataset = load_dataset(dataset_path)
        dataset = infer_classification(dataset, model_name, output_path)
    return dataset

if __name__ == "__main__":
    model_name = "llama3.2:1b"
    dataset_path = "../../data/Qualtrics_Annotations_formatB.csv"
    output_path = "./data/Qualtrics_Annotations_formatB_out_" + model_name + ".csv"
    confusion_matrix_path = "./results/confusion_matrix_" + model_name + ".png"

    dataset = get_difficulty_level(dataset_path, model_name, output_path) # infer or load the difficulty level

    print(dataset)
    # for each value of the column "difficulty", print value if not in ["Very Easy", "Easy", "Accessible", "Complex"]
    # print(dataset[~dataset["difficulty"].isin(["Very Easy", "Easy", "Accessible", "Complex"])]["difficulty"].unique())

    evaluate_classification(dataset, confusion_matrix_path) # evaluate the classification