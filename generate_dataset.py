from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle
import progressbar



def infer(dataset, model):
    # Infer the embeddings for each sentence in the dataset
    # add progressbar
    bar = progressbar.ProgressBar(maxval=len(dataset))
    i = 0
    for df in dataset:
        df["sentence1_emb"] = model.encode(df["sentence1"])
        df["sentence2_emb"] = model.encode(df["sentence2"])
        i += 1
        bar.update(i)
    bar.finish()
    return dataset

def save(dataset, savename):
    # save dataset in pickle and text format
    with open(savename + ".pkl", "wb") as f:
        pickle.dump(dataset, f)
    with open(savename + ".txt", "w") as f: # save only sentence and associated embedding
        for df in dataset:
            for i in range(len(df["sentence1"])):
                f.write(df["sentence1"][i] + "\t" + str(df["sentence1_emb"][i]) + "\n")
                f.write(df["sentence2"][i] + "\t" + str(df["sentence2_emb"][i]) + "\n")


if __name__ == "__main__":
    model =  SentenceTransformer("Lajavaness/sentence-camembert-large")

    dataset_name = "stsb_multi_mt"
    lang = "fr"

    # Loading the dataset for evaluation
    dataset_train = load_dataset(dataset_name, name=lang, split="train")
    dataset_dev = load_dataset(dataset_name, name=lang, split="dev")
    dataset_test = load_dataset(dataset_name, name=lang, split="test")

    # Infering the embeddings
    dataset_train = infer(dataset_train, model)
    dataset_dev = infer(dataset_dev, model)
    dataset_test = infer(dataset_test, model)

    # Saving the text associated with the embeddings
    save(dataset_train, dataset_name + "_" + lang + "_train")
    save(dataset_dev, dataset_name + "_" + lang + "_dev")
    save(dataset_test, dataset_name + "_" + lang + "_test")