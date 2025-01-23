from ollama import chat
from ollama import ChatResponse
import pandas as pd
import progressbar
import os

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
    # evaluate the classification
    pass

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