from ollama import chat
from ollama import ChatResponse
import pandas as pd
import progressbar
import os
import jiwer

# Function to classify text difficulty
def classify_text_difficulty(text: str) -> str:
    # Set up the chat with the classification prompt
    response: ChatResponse = chat(model='llama3.2:1b', messages=[
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

def infer_classification(dataset):
    # dataset["difficulty"] = dataset["text"].apply(classify_text_difficulty)
    # this time let's classify difficulty level with a progressbar
    bar = progressbar.ProgressBar(maxval=len(dataset))
    i = 0
    for index, row in dataset.iterrows():
        row["difficulty"] = classify_text_difficulty(row["text"])
        i += 1
        bar.update(i)
    bar.finish()

    # save in csv format
    dataset.to_csv("./data/Qualtrics_Annotations_formatB_out.csv", index=False)

    return dataset

def evaluate_classification(dataset):
    # compare column "difficulty" with "gold_score_20_label" which contains respectively "Très Facile", "Facile", "Accessible", "+Complexe" and "Very Easy", "Easy", "Accessible", "Complex"
    correct = 0
    incorrect = 0
    for index, row in dataset.iterrows():
        if row["difficulty"] not in ["Very Easy", "Easy", "Accessible", "Complex"]:
            # compute the CER between row["difficulty"] and  each of the 4 values, and update row["difficulty"] with the value with the lowest CER
            cer = [jiwer.cer(row["difficulty"][:max(len(row["difficulty"]), 30)], value) for value in ["Very Easy", "Easy", "Accessible", "Complex"]]
            row["difficulty"] = ["Very Easy", "Easy", "Accessible", "Complex"][cer.index(min(cer))]

    # convert textual values to numerical values
    dataset["difficulty"] = dataset["difficulty"].map({"Very Easy": 0, "Easy": 1, "Accessible": 2, "Complex": 3})
    dataset["gold_score_20_label"] = dataset["gold_score_20_label"].map({"Très Facile": 0, "Facile": 1, "Accessible": 2, "+Complexe": 3})

    # compute the accuracy, adjacent accuracy and macro F1
    correct = len(dataset[dataset["difficulty"] == dataset["gold_score_20_label"]])
    adjacent = len(dataset[abs(dataset["difficulty"] - dataset["gold_score_20_label"]) <= 1])
    f1 = 2 * correct / (len(dataset) + correct)
    print(f"Accuracy: {correct / len(dataset)}")
    print(f"Adjacent Accuracy: {adjacent / len(dataset)}")
    print(f"Macro F1: {f1}")

    # score for each difficulty level
    for difficulty in [0, 1, 2, 3]:
        print()
        correct = len(dataset[(dataset["difficulty"] == dataset["gold_score_20_label"]) & (dataset["difficulty"] == difficulty)])
        adjacent = len(dataset[(abs(dataset["difficulty"] - dataset["gold_score_20_label"]) <= 1) & (dataset["difficulty"] == difficulty)])
        f1 = 2 * correct / (len(dataset[dataset["gold_score_20_label"] == difficulty]) + correct)
        print(f"Difficulty: {difficulty}")
        print(f"Accuracy: {correct / len(dataset[dataset['gold_score_20_label'] == difficulty])}")
        print(f"Adjacent Accuracy: {adjacent / len(dataset[dataset['gold_score_20_label'] == difficulty])}")
        print(f"Macro F1: {f1}")

def get_difficulty_level():
    # infer if not already done
    if os.path.exists("./data/Qualtrics_Annotations_formatB_out.csv"):
        dataset = pd.read_csv("./data/Qualtrics_Annotations_formatB_out.csv")
    else:
        dataset = load_dataset()
        dataset = infer_classification(dataset)
    return dataset


if __name__ == "__main__":
    dataset = get_difficulty_level() # infer or load the difficulty level

    print(dataset)
    # for each value of the column "difficulty", print value if not in ["Very Easy", "Easy", "Accessible", "Complex"]
    # print(dataset[~dataset["difficulty"].isin(["Very Easy", "Easy", "Accessible", "Complex"])]["difficulty"].unique())

    evaluate_classification(dataset) # evaluate the classification