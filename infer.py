from ollama import chat
from ollama import ChatResponse
import pandas as pd
import progressbar

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


if __name__ == "__main__":
    text_to_classify = "Ce n'est pas l'insecte, mais un petit morceau de velours ou de taffetas noir, rond et plat ressemblant à un grain de beauté que les femmes coquettes de la haute société se collaient sur le visage ou sur le décolleté pour mettre en valeur la blancheur de leur teint ou la perfection d'une partie de leur personne."
    difficulty_level = classify_text_difficulty(text_to_classify)

    dataset = load_dataset() # csv Qualtrics_Annotations_formatB_in.csv
    print(dataset)

    dataset = infer_classification(dataset)

    print(dataset)