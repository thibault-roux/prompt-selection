from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle
import progressbar



def infer(dataset, model):
    # Add embeddings to the dataset using map
    def encode_sentences(example):
        example["sentence1_emb"] = model.encode(example["sentence1"]).tolist()  # Convert to list for serialization
        example["sentence2_emb"] = model.encode(example["sentence2"]).tolist()
        return example

    # Add a progress bar
    bar = progressbar.ProgressBar(maxval=len(dataset))
    bar.start()
    dataset = dataset.map(encode_sentences)
    bar.finish()
    return dataset

def save(dataset, savename):
    # save dataset in pickle and text format
    with open("data/" + savename + ".pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open("data/" + savename + ".txt", "w") as f: # save only sentence and associated embedding
        i = 0
        for df in dataset:
            i += 1
            f.write(df["sentence1"] + "\t" + str(df["sentence1_emb"]) + "\n")
            f.write(df["sentence2"] + "\t" + str(df["sentence2_emb"]) + "\n")
            if i > 9:
                break


if __name__ == "__main__":
    model =  SentenceTransformer("Lajavaness/sentence-camembert-large")

    dataset_name = "stsb_multi_mt"
    lang = "fr"

    # Loading the dataset for evaluation
    dataset_train = load_dataset(dataset_name, name=lang, split="train")
    dataset_dev = load_dataset(dataset_name, name=lang, split="dev")
    dataset_test = load_dataset(dataset_name, name=lang, split="test")

    # # delete all values of dataset except the first 10 elements
    # dataset_train = dataset_train.select(list(range(10)))
    # dataset_dev = dataset_dev.select(list(range(10)))
    # dataset_test = dataset_test.select(list(range(10)))

    # Infering the embeddings
    dataset_train = infer(dataset_train, model)
    dataset_dev = infer(dataset_dev, model)
    dataset_test = infer(dataset_test, model)

    # Saving the text associated with the embeddings
    save(dataset_train, dataset_name + "_" + lang + "_train")
    save(dataset_dev, dataset_name + "_" + lang + "_dev")
    save(dataset_test, dataset_name + "_" + lang + "_test")