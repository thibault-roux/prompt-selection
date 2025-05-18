def get_score_and_std(line):
    """
    This function extracts the score and standard deviation from a given line of text.
    The line is expected to be in the format: "score ± std_dev".
    """

    # Split the line into parts
    parts = line.split("±")
    
    # Extract the score and standard deviation
    score = float(parts[0].split()[1]) * 100  # Convert to percentage
    std_dev = float(parts[1].strip()) * 100  # Convert to percentage

    # Round the score and standard deviation to 2 decimal places
    score = round(score, 2)
    std_dev = round(std_dev, 2)

    return score, std_dev


def get_results(namefile):
    """
    This function reads a file containing results and calculates the accuracy, adjusted accuracy, and F1 score.
    The file is expected to be in a specific format with the first line containing the accuracy,
    the second line containing the adjusted accuracy, and the third line containing the F1 score.
    """

    with open(namefile, "r") as f:
        lines = f.readlines()

    # format is like this:
    # global_accuracy	0.5734689107059374 ± 0.03181062535138353

    accuracy, accuracy_std = get_score_and_std(lines[0])
    adj_accuracy, adj_accuracy_std = get_score_and_std(lines[1])
    f1, f1_std = get_score_and_std(lines[2])

    # Print the results
    # print(f"Accuracy: {accuracy} ± {accuracy_std}")
    # print(f"Adjusted Accuracy: {adj_accuracy} ± {adj_accuracy_std}")
    # print(f"F1 Score: {f1} ± {f1_std}")
    # input("Press Enter to continue...")
    return accuracy, adj_accuracy, f1



if __name__ == "__main__":

    model_names = ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "deepseek-r1:70b", "gemma3:27b", "qwen2.5:72b"]
    prompt_types = ["en_CECR", "fr_CECR", "fr_CECR_few_shot_cot_v2", "en_CECR_few_shot_cot_v2"]

    # format = "results_{model_name}_{prompt_type}.txt"
    
    namefiles = []
    with open("namefiles.txt", "r") as f:
        for line in f:
            namefiles.append(line.strip())
    # print(namefiles)

    for model_name in model_names:
        for prompt_type in prompt_types:
            namefile = f"results_{model_name}_{prompt_type}.txt"
            if namefile in namefiles:
                accuracy, adj_accuracy, f1 = get_results(namefile)
                print(f"Model: {model_name}, Prompt: {prompt_type}, Accuracy: {accuracy}, Adj Accuracy: {adj_accuracy}, F1: {f1}")