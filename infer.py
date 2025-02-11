from ollama import chat
from ollama import ChatResponse
import pandas as pd
import progressbar
import os
import jiwer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json


# protocol of annotations
instructs = {
  "description_niveaux_complexite": {
    "Très Facile": {
      "Concepts": "quotidien",
      "Vocabulaire": "simple et très fréquent",
      "Syntaxe": "phrases courtes et simples, ordre de base : sujet - verbe - complément",
      "Temps et modes": "principalement présent, passé composé + périphrases verbales (ex: futur proche)",
      "Chaîne référentielle": "complète",
      "Cadre spatio-temporel": "simple et linéaire",
      "Style": "pas de figure de style"
    },
    "Facile": {
      "Concepts": "quotidien, loisirs, travail",
      "Vocabulaire": "fréquent",
      "Syntaxe": "quelques phrases composées, ordre de base + compléments circonstanciels",
      "Temps et modes": "variés : principalement temps de l'indicatif + présent du subjonctif et du conditionnel",
      "Chaîne référentielle": "anaphores pronominales (tout type de pronoms)",
      "Cadre spatio-temporel": "parfois complexe mais toujours linéaire",
      "Style": "sens figurés courants + quelques figures de style (comparaison, métonymies, métaphores)"
    },
    "Accessible": {
      "Concepts": "tout type de concepts (traités de manière introductive si concept très spécialisé)",
      "Vocabulaire": "varié (sauf académique, très spécialisé, archaïque)",
      "Syntaxe": "phrases simples et composées, variété d'ordres syntaxiques",
      "Temps et modes": "presque tous les temps et modes (sauf temps complexes. ex: subjonctif imparfait)",
      "Chaîne référentielle": "anaphore et ellipses",
      "Cadre spatio-temporel": "complexe et non linéaire (mais le nombre de temps verbaux différents par phrase reste limité)",
      "Style": "sens figuré + variété de figures de style"
    },
    "+Complexe": "Au-delà du niveau 'Accessible'"
  },
  "equivalences_approximatives": {
    "Très Facile": {
      "Degré d'illettrisme": "1 - 2",
      "CECR": "A1",
      "Scolarité": "avant 'Facile'"
    },
    "Facile": {
      "Degré d'illettrisme": "2 - 3",
      "CECR": "A2",
      "Scolarité": "fin de primaire"
    },
    "Accessible": {
      "Degré d'illettrisme": "3 - 4",
      "CECR": "B1",
      "Scolarité": "fin de scolarité obligatoire (3ème)"
    },
    "+Complexe": "Au-delà du niveau 'Accessible'"
  }
}

instructs_json = json.dumps(instructs, indent=4, ensure_ascii=False)

# Few shot learning with chain of thougth
shot1 = "Les fruits et les légumes    La pomme est un fruit. L’ananas est un fruit. Le melon est un fruit. Les poires sont des fruits. (il y en a plusieurs)Les raisins sont des fruits. Les pommes sont des fruits. Les mandarines sont des fruits. Avec les pommes je prépare une tarte aux pommes. Avec les oranges je prépare un jus d’orange. Avec des fruits, je prépare une salade de fruits. Il faut peler les fruits avant de les manger. Je pèle la pomme, je pèle la poire. Il faut enlever les pépins de la pomme.      Le chou est un légume, les courgettes sont des légumes, les oignons sont des légumes, la salade est un légume, les carottes sont des légumes, les champignons sont des légumes. Le concombre est un légume. Il faut peler les légumes avant de les préparer. Il faut couper les légumes avant de les préparer. Il faut laver les légumes avant de les préparer. Avec les légumes, je prépare de la soupe. Avec les légumes, je prépare une salade. Avec les pommes de terre je prépare des frites."
cot1 = "Ce texte est de niveau Très Facile.    Justification : 1) Vocabulaire simple et courant : Les mots utilisés sont basiques et familiers. 2) Phrases courtes et structurées de manière répétitive : Cela facilite la compréhension. 3) Aucune notion abstraite ou complexe : Le texte reste concret et factuel. 4) Présence de nombreuses répétitions : Elles renforcent la compréhension et la mémorisation.    Ce type de texte convient aux jeunes enfants ou aux débutants en apprentissage du français. "
value1 = "Très Facile"

shot2 = "Les cultures en Afrique du Nord    Les trois pays du Maghreb que sont la Tunisie, le Maroc et l'Algérie ont quasiment les mêmes productions agricoles. Plus on va vers le sud, plus les cultures, les arbres et l'herbe deviennent rares en raison du manque d'eau.  Du nord au sud, la production se répartit comme suit :  - vigne, agrumes (oranges, citrons, mandarines), oliviers, légumes;  - céréales et élevage de moutons;  - dattes, dans les oasis du désert.  Mais l'importance de chaque production varie beaucoup d'un pays à l'autre. Ainsi, en Algérie, la vigne est en tête des productions; au Maroc, ce sont les céréales et l'élevage, tandis qu'en Tunisie, l'olive est prédominante."
cot2 = "Ce texte est de niveau Facile.    Justification : 1) Vocabulaire simple et accessible, avec quelques termes spécifiques mais compréhensibles dans le contexte (ex. : 'productions agricoles', 'élevage', 'oasis'). 2) Phrases courtes et bien structurées, facilitant la lecture. 3) Organisation logique des informations (du nord au sud, puis par pays). 4) Quelques comparaisons, mais elles restent simples et ne nécessitent pas une analyse approfondie.    Ce texte est donc Facile, adapté à un public ayant une maîtrise élémentaire du français. "
value2 = "Facile"

shot3 = "Horoscope de la semaine du 11 au 17 décembre 2023 pour le Bélier (21 mars - 21 avril)    À la croisée des chemins. Côté pro, si vos objectifs sont clairs, concentrez rendez-vous et prises de décision avant la Nouvelle Lune du 13. Si vous hésitez, patience. De nouvelles idées émergent mais tout est à refaire.  Le signe allié : le Capricorne, il sécurise vos prises de décisions."
cot3 = "Ce texte est de niveau Accessible.    Justification : 1) Vocabulaire relativement simple : Bien que le texte inclut des termes spécifiques comme 'prise de décision' et 'Nouvelle Lune', ceux-ci restent compréhensibles dans le contexte. 2) Idées directes et claires : Les conseils sont explicites (se concentrer avant la Nouvelle Lune, patienter si on hésite). 3) Structure logique et facile à suivre : Le texte présente des éléments consécutifs qui sont faciles à comprendre pour un public ayant un niveau de langue intermédiaire. 4) Un peu de métaphore mais sans complexité excessive : 'À la croisée des chemins' et 'sécurise vos prises de décisions' sont des expressions courantes dans les horoscopes et n'alourdissent pas le message.    Ce texte est donc Accessible, adapté à un public ayant une maîtrise moyenne du français. "
value3 = "Accessible"

shot4 = "La sensibilité écologique a connu au cours des dernières années une spectaculaire extension. Alors qu'il y a vingt ans à peine, elle paraissait être l'apanage de ceux que l'on appelait les «enfants gâtés» de la croissance, tout le monde ou presque se déclare aujourd'hui écologiste. Ou, au moins, prêt à prendre au sérieux la question de la protection de la nature, devenue «patrimoine commun» de l'humanité. Le phénomène est mondial, mais particulièrement net chez les Occidentaux, convaincus d'être menacés par les catastrophes écologiques, persuadés des dangers qui pèsent sur la planète et préoccupés par le monde qu'ils laisseront aux générations futures. Le consensus écologique concerne désormais de larges fractions de la population. Tous ceux qui font de la politique se disent «verts», les scientifiques veulent protéger la Terre, les industriels vendre du propre, les consommateurs commencer à modifier leurs comportements et les gens défendre leur cadre de vie.  Cet unanimisme est ambigu et, à l'évidence, tout le monde ne se fait pas la même idée de la nature. La sensibilité écologique s'incarne dans des clientèles, des programmes et des pratiques extrêmement variés et forme une véritable nébuleuse. Elle peut servir de cadre à ceux qui aspirent à une transformation totale de leur vie, comme à ceux qui n'y cherchent que des activités ponctuelles. Elle peut être l'occasion de nouveaux modes de consommation, d'une volonté de maintenir la diversité des milieux naturels et des cultures, etc. La recherche urgente de nouveaux rapports entre la personne et la planète peut ainsi prendre mille détours et cette variété constitue l'un des fondements de la vitalité actuelle de l'écologie.  D'après l'introduction de L'Équivoque écologique, P. Alphandéry, P. Bitoun et Y. Dupont, La Découverte, Essais, 1991."
cot4 = "Ce texte est de niveau +Complexe.    Justification : 1) Vocabulaire riche et abstrait : Des termes comme 'apanage', 'clientèles', 'nébuleuse', 'unanimisme' nécessitent une bonne maîtrise du français pour être bien compris. 2) Idées nuancées et complexes : Le texte discute des différentes facettes de la sensibilité écologique, de ses implications et de ses contradictions. Il invite à une réflexion approfondie sur le sujet. 3) Concepts philosophiques et sociétaux : Le texte aborde des questions comme la transformation de la vie, les rapports entre la personne et la planète, ce qui demande une certaine capacité d'analyse et d'abstraction. 4) Structure élaborée : Le texte est dense, avec des phrases longues et des idées qui s'entrelacent. Il nécessite une attention particulière pour saisir toutes les nuances.    Ce texte est donc +Complexe, adapté à un public ayant une bonne maîtrise du français et capable de traiter des sujets abstraits et nuancés. "
value4 = "+Complexe"



# Function to classify text difficulty
def classify_text_difficulty(text: str, model_name: str, prompt_type: str) -> str:
    global instructs_json, shot1, value1, shot2, value2, shot3, value3, shot4, value4, cot1, cot2, cot3, cot4

    if prompt_type == "en":
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'You are an expert in text difficulty classification. '
                    'Classify the given text into one of the following levels of difficulty: '
                    'Very Easy, Easy, Accessible, or Complex. '
                    'Desired format: <Level of difficulty>'
                ),
            },
            {
                'role': 'user',
                'content': text,
            },
        ])
    elif prompt_type == "fr":
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'Vous êtes un expert dans la classification de difficulté de texte. '
                    'Classifier le texte donnée selon les niveaux de difficultés suivants : '
                    'Très Facile, Facile, Accessible, ou +Complexe. '
                    'Format souhaité: <Niveau de difficulté>'
                ),
            },
            {
                'role': 'user',
                'content': text,
            },
        ])
    elif prompt_type == "en_do_not":
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'Classify the given text into one of the following levels of difficulty: '
                    'Very Easy, Easy, Accessible, or Complex. '
                    'Do not predict only <Accessible>. '
                    'Desired format: <Level of difficulty>'
                ),
            },
            {
                'role': 'user',
                'content': text,
            },
        ])
    elif prompt_type == "fr_do_not":
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'Classifiez le texte donnée selon les niveaux de difficultés suivants : '
                    'Très Facile, Facile, Accessible, ou +Complexe. '
                    'Ne prédisez pas seulement <Accessible>. '
                    'Format souhaité: <Niveau de difficulté>'
                ),
            },
            {
                'role': 'user',
                'content': text,
            },
        ])
    elif prompt_type == "fr_few_shot":
        shot1 = "Les fruits et les légumes    La pomme est un fruit. L’ananas est un fruit. Le melon est un fruit. Les poires sont des fruits. (il y en a plusieurs)Les raisins sont des fruits. Les pommes sont des fruits. Les mandarines sont des fruits. Avec les pommes je prépare une tarte aux pommes. Avec les oranges je prépare un jus d’orange. Avec des fruits, je prépare une salade de fruits. Il faut peler les fruits avant de les manger. Je pèle la pomme, je pèle la poire. Il faut enlever les pépins de la pomme.      Le chou est un légume, les courgettes sont des légumes, les oignons sont des légumes, la salade est un légume, les carottes sont des légumes, les champignons sont des légumes. Le concombre est un légume. Il faut peler les légumes avant de les préparer. Il faut couper les légumes avant de les préparer. Il faut laver les légumes avant de les préparer. Avec les légumes, je prépare de la soupe. Avec les légumes, je prépare une salade. Avec les pommes de terre je prépare des frites."
        value1 = "Très Facile"
        shot2 = "Les cultures en Afrique du Nord    Les trois pays du Maghreb que sont la Tunisie, le Maroc et l'Algérie ont quasiment les mêmes productions agricoles. Plus on va vers le sud, plus les cultures, les arbres et l'herbe deviennent rares en raison du manque d'eau.  Du nord au sud, la production se répartit comme suit :  - vigne, agrumes (oranges, citrons, mandarines), oliviers, légumes;  - céréales et élevage de moutons;  - dattes, dans les oasis du désert.  Mais l'importance de chaque production varie beaucoup d'un pays à l'autre. Ainsi, en Algérie, la vigne est en tête des productions; au Maroc, ce sont les céréales et l'élevage, tandis qu'en Tunisie, l'olive est prédominante."
        value2 = "Facile"
        shot3 = "Horoscope de la semaine du 11 au 17 décembre 2023 pour le Bélier (21 mars - 21 avril)    À la croisée des chemins. Côté pro, si vos objectifs sont clairs, concentrez rendez-vous et prises de décision avant la Nouvelle Lune du 13. Si vous hésitez, patience. De nouvelles idées émergent mais tout est à refaire.  Le signe allié : le Capricorne, il sécurise vos prises de décisions."
        value3 = "Accessible"
        shot4 = "La sensibilité écologique a connu au cours des dernières années une spectaculaire extension. Alors qu'il y a vingt ans à peine, elle paraissait être l'apanage de ceux que l'on appelait les «enfants gâtés» de la croissance, tout le monde ou presque se déclare aujourd'hui écologiste. Ou, au moins, prêt à prendre au sérieux la question de la protection de la nature, devenue «patrimoine commun» de l'humanité. Le phénomène est mondial, mais particulièrement net chez les Occidentaux, convaincus d'être menacés par les catastrophes écologiques, persuadés des dangers qui pèsent sur la planète et préoccupés par le monde qu'ils laisseront aux générations futures. Le consensus écologique concerne désormais de larges fractions de la population. Tous ceux qui font de la politique se disent «verts», les scientifiques veulent protéger la Terre, les industriels vendre du propre, les consommateurs commencer à modifier leurs comportements et les gens défendre leur cadre de vie.  Cet unanimisme est ambigu et, à l'évidence, tout le monde ne se fait pas la même idée de la nature. La sensibilité écologique s'incarne dans des clientèles, des programmes et des pratiques extrêmement variés et forme une véritable nébuleuse. Elle peut servir de cadre à ceux qui aspirent à une transformation totale de leur vie, comme à ceux qui n'y cherchent que des activités ponctuelles. Elle peut être l'occasion de nouveaux modes de consommation, d'une volonté de maintenir la diversité des milieux naturels et des cultures, etc. La recherche urgente de nouveaux rapports entre la personne et la planète peut ainsi prendre mille détours et cette variété constitue l'un des fondements de la vitalité actuelle de l'écologie.  D'après l'introduction de L'Équivoque écologique, P. Alphandéry, P. Bitoun et Y. Dupont, La Découverte, Essais, 1991."
        value4 = "+Complexe"
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'Vous êtes un expert dans la classification de difficulté de texte. '
                    'Classifier le texte donnée selon les niveaux de difficultés suivants : '
                    'Très Facile, Facile, Accessible, ou +Complexe. '
                ),
            },
            {'role': 'user','content': shot1,},
            {'role': 'assistant', 'content': "<" + value1 + ">",},
            {'role': 'user', 'content': shot2,},
            {'role': 'assistant', 'content': "<" + value2 + ">",},
            {'role': 'user', 'content': shot3,},
            {'role': 'assistant', 'content': "<" + value3 + ">",},
            {'role': 'user', 'content': shot4,},
            {'role': 'assistant', 'content': "<" + value4 + ">",},
            {
                'role': 'user',
                'content': text,
            },
        ])
    elif prompt_type == "fr_few_shot_cot": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'Vous êtes un expert dans la classification de difficulté de texte. '
                    'Classifiez le texte donnée selon les niveaux de difficultés suivants : '
                    'Très Facile, Facile, Accessible, ou +Complexe. '
                ),
            },
            {'role': 'user','content': shot1,},
            {'role': 'assistant', 'content': cot1 + "<" + value1 + ">",},
            {'role': 'user', 'content': shot2,},
            {'role': 'assistant', 'content': cot2 + "<" + value2 + ">",},
            {'role': 'user', 'content': shot3,},
            {'role': 'assistant', 'content': cot3 + "<" + value3 + ">",},
            {'role': 'user', 'content': shot4,},
            {'role': 'assistant', 'content': cot4 + "<" + value4 + ">",},
            {
                'role': 'user',
                'content': text,
            },
        ])
    elif prompt_type == "fr_few_shot_cot_with_protocol": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'Vous êtes un expert dans la classification de difficulté de texte. '
                    'Classifiez le texte donné selon les niveaux de difficulté suivants : '
                    'Très Facile, Facile, Accessible, ou +Complexe.\n\n'
                    'Voici la description détaillée des niveaux de difficulté :\n'
                    f'{instructs_json}'
                ),
            },
            {'role': 'user','content': shot1,},
            {'role': 'assistant', 'content': cot1 + "\nNiveau: **" + value1 + "**",},
            {'role': 'user', 'content': shot2,},
            {'role': 'assistant', 'content': cot2 + "\nNiveau: **" + value2 + "**",},
            {'role': 'user', 'content': shot3,},
            {'role': 'assistant', 'content': cot3 + "\nNiveau: **" + value3 + "**",},
            {'role': 'user', 'content': shot4,},
            {'role': 'assistant', 'content': cot4 + "\nNiveau: **" + value4 + "**",},
            {
                'role': 'user',
                'content': text,
            },
        ])
    else:
        raise ValueError("Invalid prompt type. Must be 'en', 'fr', 'en_do_not', 'fr_do_not', 'fr_few_shot', or 'fr_few_shot_cot'.")
    return response['message']['content']




def load_dataset(path="../../data/Qualtrics_Annotations_formatB.csv"):
    df = pd.read_csv(path)
    return df

def infer_classification(dataset, model_name, prompt_type, csv_path):
    bar = progressbar.ProgressBar(maxval=len(dataset))
    i = 0
    for index, row in dataset.iterrows():
        dataset.at[index, "difficulty"] = classify_text_difficulty(row["text"], model_name, prompt_type)
        i += 1
        bar.update(i)
    bar.finish()
    # save in csv format
    dataset.to_csv(csv_path, index=False)
    return dataset

def save_confusion_matrix(y_true, y_pred, confusion_matrix_path): # csv_path not used
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Très Facile", "Facile", "Accessible", "+Complexe"],
                yticklabels=["Très Facile", "Facile", "Accessible", "+Complexe"])
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités terrain")
    plt.title("Matrice de Confusion")
    plt.show()

    # save
    plt.savefig(confusion_matrix_path)


def evaluate_classification(dataset, confusion_matrix_path, results_path):
    pattern = r"(?:<|\*\*)(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe)(?:>|\*\*)"

    # Correction des valeurs erronées dans la colonne "difficulty"
    for index, row in dataset.iterrows():
        if row["difficulty"] not in ["Very Easy", "Easy", "Accessible", "Complex", "Très Facile", "Facile", "Accessible", "+Complexe"]:
            # print("Text:", row["text"])
            # print("Before:", row["difficulty"])

            match = re.search(pattern, row["difficulty"])
            if match:
                predicted_class = match.group(1)
                dataset.at[index, "difficulty"] = predicted_class
            else:
                match = re.search(r"(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe)", row["difficulty"][-35:])
                if match:
                    predicted_class = match.group(1)
                    dataset.at[index, "difficulty"] = predicted_class
                else:
                    # Calcul du CER pour chaque valeur candidate et sélection de la meilleure
                    candidates = ["Very Easy", "Easy", "Accessible", "Complex", "Très Facile", "Facile", "Accessible", "+Complexe"]
                    # cer_scores = [jiwer.cer(row["difficulty"][:max(len(row["difficulty"]), 30)], candidate) for candidate in candidates]
                    cer_scores = [jiwer.cer(row["difficulty"][-15:].lower(), candidate.lower()) for candidate in candidates]
                    dataset.at[index, "difficulty"] = candidates[cer_scores.index(min(cer_scores))]
            # print("After:", dataset.at[index, "difficulty"])
            # print("Real:", row["gold_score_20_label"])
            # input()

    # Conversion des valeurs textuelles en numériques
    mapping_pred = {"Very Easy": 0, "Easy": 1, "Accessible": 2, "Complex": 3, "Très Facile": 0, "Facile": 1, "Accessible": 2, "+Complexe": 3}
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

    txt = f"global_accuracy\t{global_accuracy}\nglobal_adjacent_accuracy\t{global_adjacent_accuracy}\nglobal_macro_f1\t{global_macro_f1}\n"

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

        txt += f"difficulty_{difficulty}_accuracy\t{class_accuracy}\ndifficulty_{difficulty}_adjacent_accuracy\t{class_adjacent_accuracy}\ndifficulty_{difficulty}_f1\t{class_f1}\n"

    save_confusion_matrix(y_true, y_pred, confusion_matrix_path)
    with open(results_path, "w") as f:
        f.write(txt)

def get_difficulty_level(dataset_path, model_name, prompt_type, csv_path):
    if os.path.exists(csv_path):
        dataset = pd.read_csv(csv_path)
    else:
        dataset = load_dataset(dataset_path)
        dataset = infer_classification(dataset, model_name, prompt_type, csv_path)
    return dataset

if __name__ == "__main__":
    model_name = "deepseek-r1:70b" # "llama3.2:1b" # "deepseek-r1:70b" # "deepseek-r1:7b" # "llama3.2:1b"
    prompt_type = "fr_few_shot_cot_with_protocol" # "fr_few_shot_cot" # "fr_few_shot" # "fr_do_not" # "en_do_not" # "en" "fr"
    dataset_path = "../../data/Qualtrics_Annotations_formatB.csv"
    csv_path = "./data/Qualtrics_Annotations_formatB_out_" + model_name + "_" + prompt_type + ".csv"
    confusion_matrix_path = "./results/confusion_matrix_" + model_name + "_" + prompt_type + ".png"
    results_path = "./results/results_" + model_name + "_" + prompt_type + ".txt"

    dataset = get_difficulty_level(dataset_path, model_name, prompt_type, csv_path) # infer or load the difficulty level

    print(dataset)
    # for each value of the column "difficulty", print value if not in ["Very Easy", "Easy", "Accessible", "Complex"]
    # print(dataset[~dataset["difficulty"].isin(["Very Easy", "Easy", "Accessible", "Complex"])]["difficulty"].unique())

    evaluate_classification(dataset, confusion_matrix_path, results_path) # evaluate the classification