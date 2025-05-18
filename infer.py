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
import numpy as np


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

if True:
    # Few shot learning with chain of thougth
    shot1 = "Les fruits et les légumes    La pomme est un fruit. L’ananas est un fruit. Le melon est un fruit. Les poires sont des fruits. (il y en a plusieurs)Les raisins sont des fruits. Les pommes sont des fruits. Les mandarines sont des fruits. Avec les pommes je prépare une tarte aux pommes. Avec les oranges je prépare un jus d’orange. Avec des fruits, je prépare une salade de fruits. Il faut peler les fruits avant de les manger. Je pèle la pomme, je pèle la poire. Il faut enlever les pépins de la pomme.      Le chou est un légume, les courgettes sont des légumes, les oignons sont des légumes, la salade est un légume, les carottes sont des légumes, les champignons sont des légumes. Le concombre est un légume. Il faut peler les légumes avant de les préparer. Il faut couper les légumes avant de les préparer. Il faut laver les légumes avant de les préparer. Avec les légumes, je prépare de la soupe. Avec les légumes, je prépare une salade. Avec les pommes de terre je prépare des frites."
    cot1 = "Ce texte est de niveau Très Facile.    Justification : 1) Vocabulaire simple et courant : Les mots utilisés sont basiques et familiers. 2) Phrases courtes et structurées de manière répétitive : Cela facilite la compréhension. 3) Aucune notion abstraite ou complexe : Le texte reste concret et factuel. 4) Présence de nombreuses répétitions : Elles renforcent la compréhension et la mémorisation.    Ce type de texte convient aux jeunes enfants ou aux débutants en apprentissage du français. "
    cot1_CECR = "Ce texte est de niveau A1.    Justification : 1) Vocabulaire simple et courant : Les mots utilisés sont basiques et familiers. 2) Phrases courtes et structurées de manière répétitive : Cela facilite la compréhension. 3) Aucune notion abstraite ou complexe : Le texte reste concret et factuel. 4) Présence de nombreuses répétitions : Elles renforcent la compréhension et la mémorisation.    Ce type de texte convient aux jeunes enfants ou aux débutants en apprentissage du français."
    value1 = "Très Facile"

    shot2 = "Les cultures en Afrique du Nord    Les trois pays du Maghreb que sont la Tunisie, le Maroc et l'Algérie ont quasiment les mêmes productions agricoles. Plus on va vers le sud, plus les cultures, les arbres et l'herbe deviennent rares en raison du manque d'eau.  Du nord au sud, la production se répartit comme suit :  - vigne, agrumes (oranges, citrons, mandarines), oliviers, légumes;  - céréales et élevage de moutons;  - dattes, dans les oasis du désert.  Mais l'importance de chaque production varie beaucoup d'un pays à l'autre. Ainsi, en Algérie, la vigne est en tête des productions; au Maroc, ce sont les céréales et l'élevage, tandis qu'en Tunisie, l'olive est prédominante."
    cot2 = "Ce texte est de niveau Facile.    Justification : 1) Vocabulaire simple et accessible, avec quelques termes spécifiques mais compréhensibles dans le contexte (ex. : 'productions agricoles', 'élevage', 'oasis'). 2) Phrases courtes et bien structurées, facilitant la lecture. 3) Organisation logique des informations (du nord au sud, puis par pays). 4) Quelques comparaisons, mais elles restent simples et ne nécessitent pas une analyse approfondie.    Ce texte est donc Facile, adapté à un public ayant une maîtrise élémentaire du français. "
    cot2_CECR = "Ce texte est de niveau A2.    Justification : 1) Vocabulaire simple et accessible, avec quelques termes spécifiques mais compréhensibles dans le contexte (ex. : 'productions agricoles', 'élevage', 'oasis'). 2) Phrases courtes et bien structurées, facilitant la lecture. 3) Organisation logique des informations (du nord au sud, puis par pays). 4) Quelques comparaisons, mais elles restent simples et ne nécessitent pas une analyse approfondie.    Ce texte est donc de niveau A2, adapté à un public ayant une maîtrise élémentaire du français."
    value2 = "Facile"

    shot3 = "Horoscope de la semaine du 11 au 17 décembre 2023 pour le Bélier (21 mars - 21 avril)    À la croisée des chemins. Côté pro, si vos objectifs sont clairs, concentrez rendez-vous et prises de décision avant la Nouvelle Lune du 13. Si vous hésitez, patience. De nouvelles idées émergent mais tout est à refaire.  Le signe allié : le Capricorne, il sécurise vos prises de décisions."
    cot3 = "Ce texte est de niveau Accessible.    Justification : 1) Vocabulaire relativement simple : Bien que le texte inclut des termes spécifiques comme 'prise de décision' et 'Nouvelle Lune', ceux-ci restent compréhensibles dans le contexte. 2) Idées directes et claires : Les conseils sont explicites (se concentrer avant la Nouvelle Lune, patienter si on hésite). 3) Structure logique et facile à suivre : Le texte présente des éléments consécutifs qui sont faciles à comprendre pour un public ayant un niveau de langue intermédiaire. 4) Un peu de métaphore mais sans complexité excessive : 'À la croisée des chemins' et 'sécurise vos prises de décisions' sont des expressions courantes dans les horoscopes et n'alourdissent pas le message.    Ce texte est donc Accessible, adapté à un public ayant une maîtrise moyenne du français. "
    cot3_CECR = "Ce texte est de niveau B1.    Justification : 1) Vocabulaire relativement simple : Bien que le texte inclut des termes spécifiques comme 'prise de décision' et 'Nouvelle Lune', ceux-ci restent compréhensibles dans le contexte. 2) Idées directes et claires : Les conseils sont explicites (se concentrer avant la Nouvelle Lune, patienter si on hésite). 3) Structure logique et facile à suivre : Le texte présente des éléments consécutifs qui sont faciles à comprendre pour un public ayant un niveau de langue intermédiaire. 4) Un peu de métaphore mais sans complexité excessive : 'À la croisée des chemins' et 'sécurise vos prises de décisions' sont des expressions courantes dans les horoscopes et n'alourdissent pas le message.    Ce texte est donc de niveau B1, adapté à un public ayant une maîtrise moyenne du français."
    value3 = "Accessible"

    shot4 = "La sensibilité écologique a connu au cours des dernières années une spectaculaire extension. Alors qu'il y a vingt ans à peine, elle paraissait être l'apanage de ceux que l'on appelait les «enfants gâtés» de la croissance, tout le monde ou presque se déclare aujourd'hui écologiste. Ou, au moins, prêt à prendre au sérieux la question de la protection de la nature, devenue «patrimoine commun» de l'humanité. Le phénomène est mondial, mais particulièrement net chez les Occidentaux, convaincus d'être menacés par les catastrophes écologiques, persuadés des dangers qui pèsent sur la planète et préoccupés par le monde qu'ils laisseront aux générations futures. Le consensus écologique concerne désormais de larges fractions de la population. Tous ceux qui font de la politique se disent «verts», les scientifiques veulent protéger la Terre, les industriels vendre du propre, les consommateurs commencer à modifier leurs comportements et les gens défendre leur cadre de vie.  Cet unanimisme est ambigu et, à l'évidence, tout le monde ne se fait pas la même idée de la nature. La sensibilité écologique s'incarne dans des clientèles, des programmes et des pratiques extrêmement variés et forme une véritable nébuleuse. Elle peut servir de cadre à ceux qui aspirent à une transformation totale de leur vie, comme à ceux qui n'y cherchent que des activités ponctuelles. Elle peut être l'occasion de nouveaux modes de consommation, d'une volonté de maintenir la diversité des milieux naturels et des cultures, etc. La recherche urgente de nouveaux rapports entre la personne et la planète peut ainsi prendre mille détours et cette variété constitue l'un des fondements de la vitalité actuelle de l'écologie.  D'après l'introduction de L'Équivoque écologique, P. Alphandéry, P. Bitoun et Y. Dupont, La Découverte, Essais, 1991."
    cot4 = "Ce texte est de niveau +Complexe.    Justification : 1) Vocabulaire riche et abstrait : Des termes comme 'apanage', 'clientèles', 'nébuleuse', 'unanimisme' nécessitent une bonne maîtrise du français pour être bien compris. 2) Idées nuancées et complexes : Le texte discute des différentes facettes de la sensibilité écologique, de ses implications et de ses contradictions. Il invite à une réflexion approfondie sur le sujet. 3) Concepts philosophiques et sociétaux : Le texte aborde des questions comme la transformation de la vie, les rapports entre la personne et la planète, ce qui demande une certaine capacité d'analyse et d'abstraction. 4) Structure élaborée : Le texte est dense, avec des phrases longues et des idées qui s'entrelacent. Il nécessite une attention particulière pour saisir toutes les nuances.    Ce texte est donc +Complexe, adapté à un public ayant une bonne maîtrise du français et capable de traiter des sujets abstraits et nuancés. "
    cot4_CECR = "Ce texte est de niveau B2.    Justification : 1) Vocabulaire riche et abstrait : Des termes comme 'apanage', 'clientèles', 'nébuleuse', 'unanimisme' nécessitent une bonne maîtrise du français pour être bien compris. 2) Idées nuancées et complexes : Le texte discute des différentes facettes de la sensibilité écologique, de ses implications et de ses contradictions. Il invite à une réflexion approfondie sur le sujet. 3) Concepts philosophiques et sociétaux : Le texte aborde des questions comme la transformation de la vie, les rapports entre la personne et la planète, ce qui demande une certaine capacité d'analyse et d'abstraction. 4) Structure élaborée : Le texte est dense, avec des phrases longues et des idées qui s'entrelacent. Il nécessite une attention particulière pour saisir toutes les nuances.    Ce texte est donc de niveau B2, adapté à un public ayant une bonne maîtrise du français et capable de traiter des sujets abstraits et nuancés."
    value4 = "+Complexe"

if True:
    # Few shot learning with chain of thought V2
    shot1_v2 = "Le coq est mort    Le coq est mort,  Le coq est mort,    Le coq est mort.    Il ne dira plus cocodi, cocoda  Il ne dira plus cocodi, cocoda,  cocodicodi, codicoda  cocodicodi, codicoda"
    cot1_v2 = "Le texte fourni est une répétition de phrases très simples et courtes, utilisant un vocabulaire élémentaire et des structures grammaticales basiques. L'usage est limité à des énoncés descriptifs sans complexité syntaxique ni nuances lexicales. Ce type de formulation correspond au niveau élémentaire de compréhension et d'expression. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets."
    value1_v2 = "Très Facile"

    shot2_v2 = "Inscription à la médiathèque    BULLETIN D'INSCRIPTION  Nom: ....................  Prénom: .................  Date de Naissance :.........Sexe : F / M  Adresse:...................  Code postal:....... Ville : ............  Téléphone portable :......................  Téléphone fixe :..........................  Email:....................  L'email sera utilisé pour vous informer de la mise à disposition de vos réservations.  J'autorise le réseau des médiathèques à me contacter par :  Téléphone : oui non  Email : oui non  Je garantis sur l'honneur l'exactitude des renseignements ci-dessus.  En signant ce document, je reconnais avoir pris connaissance du règlement intérieur et accepte l'ensemble de ces clauses.    Lieu, date, signature"
    cot2_v2 = "Le texte fourni est un formulaire d'inscription à une médiathèque. Il se compose principalement de champs à remplir avec des informations personnelles et d'instructions simples et formelles. La structure est claire et dépourvue de complexités syntaxiques ou lexicales. Aucun argument développé ni usage avancé de la langue n'est requis pour le comprendre. Selon le Cadre européen commun de référence pour les langues (CECRL), ce type de texte correspond à un niveau A2, car il nécessite une compréhension de phrases courantes et de formulaires administratifs simples."
    value2_v2 = "Facile"

    shot3_v2 = "Quelqu'un de bien    Debout devant ses illusions  Une femme que plus rien ne dérange  Détenue de son abandon  Son ennui lui donne le change    Que retient-elle de sa vie  Qu'elle pourrait revoir en peinture  Dans un joli cadre verni  En évidence sur un mur    Un mariage en Technicolor  Un couple dans les tons pastel  Assez d'argent sans trop d'efforts  Pour deux trois folies mensuelles    Elle a rêvé comme tout le monde  Qu'elle tutoierait quelques vedettes  Mais ses rêves en elle se fondent  Maintenant son espoir serait d'être    Refrain  Juste quelqu'un de bien  Quelqu'un de bien  Le cœur à portée de main  Juste quelqu'un de bien  Sans grand destin  Une amie à qui l'on tient  Juste quelqu'un de bien  Quelqu'un de bien    Il m'arrive aussi de ces heures  Où ma vie se penche sur le vide  Coupés tous les bruits du moteur  Au-dessus de terres arides    Je plane à l'aube d'un malaise  Comme un soleil qui veut du mal  Aucune réponse n'apaise  Mes questions à la verticale    J'dis bonjour à la boulangère  Je tiens la porte à la vieille dame  Des fleurs pour la fête des mères  Et ce week-end à Amsterdam    Pour que tu m'aimes encore un peu  Quand je n'attends que du mépris  À l'heure où s'enfuit le Bon Dieu  Qui pourrait me dire si je suis    Au refrain    J'aime à penser que tous les hommes  S'arrêtent parfois de poursuivre  L'ambition de marcher sur Rome  Et connaissent la peur de vivre    Sur le bas-côté de la route  Sur la bande d'arrêt d'urgence  Comme des gens qui parlent et qui doutent  D'être au-delà des apparences    Au refrain    Interprète : Enzo Enzo  Paroles et musique : Kent. D.R., 1994."
    cot3_v2 = "Le texte présente une narration poétique avec des structures variées et un vocabulaire accessible, bien que parfois nuancé. Les thèmes abordés (désillusions, quête de bonheur, gestes du quotidien) restent compréhensibles pour un apprenant intermédiaire, sans nécessiter une analyse linguistique approfondie. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau B1, car il exige une compréhension de descriptions, d’émotions et d’événements dans un langage relativement courant, tout en intégrant quelques figures de style accessibles."
    value3_v2 = "Accessible"

    shot4_v2 = "Monsieur Charles Picqué, Bourgmestre de Saint-Gilles,   Madame Martine Wille, Bourgmestre f. f.   Madame Catherine François, Présidente,   Monsieur Thierry Van Campenhout, Directeur,   et l'équipe du Centre culturel Jacques Franck,    LA TOPOGRAPHIE DU SIGNE  Nicole Callebaut   Peinture/Dessin    L'œuvre picturale de Nicole Callebaut privilégie la suggestion. (...) Parfois dessins, photos et objets se conjuguent en installations et apprivoisent les éléments : « L'air, l'eau et la terre », cartographie décalée, cristallisation sombre, tempêtes mouvementées. (...) Poursuivant sa recherche obstinée, Nicole Callebaut explore matières et techniques, et nous offre une œuvre ouverte, à la ferveur unique qui permet à chaque regardeur de forger son propre décodage. A contempler donc pour choisir un voyage, une dérive, un chemin de connaissance personnel. La poésie ici a rendez-vous avec un frémissement subtil.  (Jo Dustin/Septembre 2010)    Vernissage le vendredi 4 février 2011, de 18h à 21h  Exposition du 5 février au 20 mars 2011  Du mardi au vendredi de 11h à 18h30, les samedis de 11h à 13h30 et de 14h à 18h30 & les dimanches de 14h à 17h et de 19h à 22h. (Entrée libre)   Centre culturel Jacques Franck / direction : Thierry Van Campenhout Chaussée de Waterloo, 94 - 1060 Bruxelles  tél: 02 538 90 20 - email: infoccjf@brutele.be www.ccjacquesfranck.be   Avec le soutien de la Commune de Saint-Gilles, la Communauté française de Belgique, la Commission communautaire française, et la Loterie Nationale."
    cot4_v2 = "Le texte est une annonce d'exposition mêlant des éléments informatifs (dates, lieu, organisateurs) et une description artistique du travail de Nicole Callebaut. La partie descriptive utilise un langage relativement abstrait et poétique avec des expressions comme « cartographie décalée », « cristallisation sombre », et « frémissement subtil », ce qui demande une certaine aisance en français pour en saisir les nuances. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau B2, car il nécessite une bonne compréhension des descriptions subjectives et de la terminologie artistique, tout en restant accessible aux apprenants avancés."
    value4_v2 = "+Complexe"

if True:
    # Few shot learning with chain of thought V3
    shot1_v3 = "Salut Martin, Est-ce que tu peux venir me chercher en face de la gare ? Je me suis fait arrêter par la police et j'ai perdu tous mes points, Il faut que tu conduises la voiture, Merci de me rejoindre le plus vite possible, Damien"
    cot1_v3 = "Le texte est un message simple et direct, rédigé dans un langage courant avec des phrases courtes et une structure grammaticale basique. Il utilise un vocabulaire élémentaire et des expressions du quotidien sans complexité particulière. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau A1, car il repose sur des phrases simples et des notions essentielles de communication."
    value1_v3 = "Très Facile"

    shot2_v3 = "A L'ATTENTION DE TOUT LE PERSONNEL  Mardi 11/10/2022  Bonjour,  La direction organise une réunion mercredi 19 octobre, à 15h pour tout le personnel afin de présenter les nouvelles formations, Inscriptions au bureau de Céline avant vendredi 18h,  Merci de votre participation,  La direction"
    cot2_v3 = "Le texte est une note d'information interne rédigée avec des phrases courtes et une structure simple. Il utilise un vocabulaire basique et des formulations claires, typiques d’une communication fonctionnelle. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau A2, car il demande une compréhension de consignes écrites et d’informations factuelles simples, sans nécessiter de compétences linguistiques avancées."
    value2_v3 = "Facile"

    shot3_v3 = "Infos horaires train Saintes-Niort    Nombre de trajets par jour        10  Durée moyenne d’un trajet       01h34  Durée du trajet le plus court     01h09  Première heure de départ        06h05  Dernière heure de départ         18h10    Horaires train Saintes-Niort Mercredi 13 décembre 2023  Départ        Arrivée        Durée     Transport  06h05         07h22         01h17     TER                        direct  06h56         08h13         01h17     TER                        direct  07h13         09h25         02h12     TER, TGV INOUI    1 correspondance      09h25         11h12         01h47     INTERCITÉS, TER  1 correspondance      10h06         11h15         01h09     TER                        direct    source : www,sncf-connect,com"
    cot3_v3 = "Le texte est une fiche d’informations sur les horaires de train, présentant des données factuelles sous forme de liste et de tableau. Il utilise un vocabulaire courant lié aux transports et demande une capacité de lecture pour extraire des informations précises (horaires, durées, correspondances). Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau A2, car il nécessite une compréhension d’informations simples mais détaillées, telles que des horaires et des trajets, ce qui dépasse le niveau élémentaire."
    value3_v3 = "Facile"

    shot4_v3 = "Débats - Médias traditionnels, médias sociaux,    Guillaume  On a souvent tendance à opposer les médias traditionnels et les réseaux sociaux, Pourtant, les réseaux sociaux témoignent des mêmes défauts que les médias traditionnels : ils cherchent tous les deux à obtenir des revenus de la publicité,  Cependant, une chose fait toute la différence : sur les réseaux sociaux, on peut échanger, on peut remettre l'info en question, Elle devient coconstruite par les différents utilisateurs,    Lionel  Pas question de réglementer les réseaux sociaux ! En leur demandant de différencier les « fake news » des « real news », nous donnerions à Facebook et Google le pouvoir de contrôler ce qui est vrai et ce qui ne l'est pas,  On peut comprendre l'intention, mais quand même, il vaut mieux que nous soyons sans cesse exposés à des informations qui s'opposent, à des nouvelles qui remettent nos croyances en question,    Christophe  Plusieurs présidents dans le monde ont été élus malgré des campagnes très dures de la part des médias d'information, Bien que les journalistes les aient accusés de contradictions et de mensonges, les électeurs ont voté pour eux, Ils n'ont pas été influencés par ces informations et ont montré que ça leur était égal,  La fameuse influence dont tout le monde parle à l'époque des réseaux sociaux a choisi son camp et a déserté les médias d'information, La victoire de ces présidents est la défaite historique des médias traditionnels"
    cot4_v3 = "Le texte est un débat sur les médias traditionnels et les réseaux sociaux, exprimé à travers trois points de vue argumentés. Il utilise des structures variées (oppositions, concessions, affirmations nuancées) et un vocabulaire relatif aux médias et à la communication. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau B1, car il demande une capacité à comprendre des opinions, à suivre une argumentation simple et à identifier les idées principales sur un sujet d’actualité."
    value4_v3 = "Accessible"

    shot5_v3 = "Les trois vérités de Bouc    (Conte du Sénégal)    Un jour, Bouc, séduit par la religion musulmane se convertit à l’islam, Il décida de se rendre à la Mecque, en pèlerinage,  Il partit, Il marcha, il marcha, et il tomba Ratch sur Hyène, Alors Hyène lui demanda :  – Eh, Bouc ! Où vas-tu donc ainsi, tout seul ?  Il répondit :  – Eh bien, je vais à la Mecque, Je suis converti à l’Islam,  Hyène lui dit :  – Dans ce cas tu es bien arrivé, La Mecque c’est ici,    Devinant ses intentions, Bouc le supplia et dit :  – De grâce, épargne-moi, Je suis père de famille,  Hyène leva le museau, éternua et lui dit :  – Tu ne partiras pas d’ici sans me dire trois vérités indiscutables,  Bouc réfléchit un moment et lui dit :  – Ah oui ?  Hyène répondit :  – Absolument, Avant de partir d’ici, tu me diras trois vérités que personne ne pourra remettre en cause,  Bouc lui dit :  – Oncle Hyène, si j’étais convaincu qu’en prenant ce chemin j’allais à ta rencontre, Dieu sait que je ne l’aurais jamais pris,  Hyène resta interdite un moment et lui dit :  – Tu as raison, Une,  Bouc réfléchit à nouveau et dit :  – Si je rentre au village, et déclare que j’ai rencontré l’hyène dans la brousse, l’on me traitera de menteur,  Hyène lui dit :  – Tu as encore raison, Deux, Il reste une vérité,  Bouc déclara :  – Je suis en tout cas certain d’une chose,    Hyène demanda :  – Laquelle ?  Bouc dit :  – Toute cette palabre, c’est parce que tu n’as pas faim,  Hyène dit :  – C'est juste ! Tu peux donc partir,  Bouc s’enfuit et sauva sa vie"
    cot5_v3 = "Le texte est un conte traditionnel du Sénégal avec une structure narrative claire, des dialogues simples et un vocabulaire accessible. Il utilise des répétitions et un schéma classique de conte avec une morale implicite. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau B1, car il nécessite une compréhension de récits structurés, de dialogues et d’expressions figurées tout en restant globalement accessible à un lecteur intermédiaire."
    value5_v3 = "Accessible"

    shot6_v3 = "Les rêveries de Madame Bovary,    Nous sommes au milieu du XIXe siècle, Emma, la fille d'un fermier de Normandie, vient d'épouser Charles Bovary, le médecin du village,    Elle songeait quelquefois que c'étaient là pourtant les plus beaux jours de sa vie, la lune de miel, comme on disait, Pour en goûter la douceur, il eût fallu, sans doute, s'en aller vers ces pays à noms sonores où les lendemains de mariage ont de plus suaves paresses ! Dans des chaises de poste, sous des stores de soie bleue, on monte au pas des routes escarpées, écoutant la chanson du postillon, qui se répète dans la montagne avec les clochettes des chèvres et le bruit sourd de la cascade, Quand le soleil se couche, on respire au bord des golfes le parfum des citronniers ; puis, le soir, sur la terrasse des villas, seuls et les doigts confondus, on regarde les étoiles en faisant des projets, Il lui semblait que certains lieux sur la terre devaient produire du bonheur, comme une plante particulière au sol et qui pousse mal tout autre part, Que ne pouvait-elle s'accouder sur le balcon des chalets suisses ou enfermer sa tristesse dans un cottage écossais, avec un mari vêtu d'un habit de velours noir [,,,] et qui porte des bottes molles, un chapeau pointu et des manchettes !    Gustave Flaubert, Madame Bovary, 1857"
    cot6_v3 = "Le texte est un extrait de Madame Bovary de Gustave Flaubert, un roman du XIXe siècle connu pour son style littéraire sophistiqué et son usage du discours indirect libre. L'extrait présente des phrases longues et descriptives, un vocabulaire riche et des structures complexes impliquant des nuances subtiles dans les pensées et émotions du personnage. Selon le Cadre européen commun de référence pour les langues (CECRL), ce texte correspond au niveau B2, car il nécessite une très bonne compréhension du français littéraire, une capacité à suivre des descriptions détaillées et à interpréter des idées abstraites."
    value6_v3 = "+Complexe"

classe2CECR = {"Très Facile": "A1", "Facile": "A2", "Accessible": "B1", "+Complexe": "B2"}
CECR2classe = {"A1": "Très Facile", "A2": "Facile", "B1": "Accessible", "B2": "+Complexe", "C1": "+Complexe", "C2": "+Complexe"}

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
    elif prompt_type == "en_CECR": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'You are a linguistic expert specialized in evaluating French language levels according to the Common European Framework of Reference for Languages (CEFR). Your task is to classify the following French text into one of the CEFR levels: A1, A2, B1, B2, C1, or C2. Respond ONLY with the most appropriate level label, without any explanation or additional text.\n'
                    '\nExample:'
                    'Text to classify: "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'CECR Level: **A1**'
                ),
            },
            {'role': 'user','content': "Classify this French text:\n" + text,},
            {'role': 'assistant', 'content': 'CECR Level: **'}
        ])
    elif prompt_type == "fr_CECR": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    "Vous êtes un expert linguistique spécialisé dans l'évaluation des niveaux de français selon le Cadre européen commun de référence pour les langues (CECR). Votre tâche consiste à classer le texte français suivant dans l'un des niveaux du CECR : A1, A2, B1, B2, C1 ou C2.\n"
                    '\nExemple :'
                    'Texte à classifier : "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'Niveau CECR: **A1**'
                ),
            },
            {'role': 'user','content': "Classifiez ce texte français :\n" + text,},
            {'role': 'assistant', 'content': 'Niveau CECR : **'}
        ])
    elif prompt_type == "en_CECR_few_shot_cot": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'You are a linguistic expert specialized in evaluating French language levels according to the Common European Framework of Reference for Languages (CEFR). Your task is to classify the following French text into one of the CEFR levels: A1, A2, B1, B2, C1, or C2.\n'
                    '\nExample:'
                    'Text to classify: "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'CECR Level: **A1**'
                ),
            },
            {'role': 'user','content': "Classify this French text:\n" + shot1,},
            {'role': 'assistant', 'content': cot1_CECR + "\n" + "CECR Level: **" + classe2CECR[value1] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot2,},
            {'role': 'assistant', 'content': cot2_CECR + "\n" + "CECR Level: **" + classe2CECR[value2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot3,},
            {'role': 'assistant', 'content': cot3_CECR + "\n" + "CECR Level: **" + classe2CECR[value3] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot4,},
            {'role': 'assistant', 'content': cot4_CECR + "\n" + "CECR Level: **" + classe2CECR[value4] + "**"},
            {'role': 'user','content': "Classify this French text:\n" + text,},
            {'role': 'assistant', 'content': 'CECR Level: **'}
        ])
    elif prompt_type == "en_CECR_few_shot_cot_v2": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    'You are a linguistic expert specialized in evaluating French language levels according to the Common European Framework of Reference for Languages (CEFR). Your task is to classify the following French text into one of the CEFR levels: A1, A2, B1, B2, C1, or C2.\n'
                    '\nExample:'
                    'Text to classify: "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'CECR Level: **A1**'
                ),
            },
            {'role': 'user', 'content': "Classify this French text:\n" + shot3_v2,},
            {'role': 'assistant', 'content': cot3_v2 + "\n" + "CECR Level: **" + classe2CECR[value3_v2] + "**"},
            {'role': 'user','content': "Classify this French text:\n" + shot1_v2,},
            {'role': 'assistant', 'content': cot1_v2 + "\n" + "CECR Level: **" + classe2CECR[value1_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot2_v2,},
            {'role': 'assistant', 'content': cot2_v2 + "\n" + "CECR Level: **" + classe2CECR[value2_v2] + "**"},
            {'role': 'user', 'content': "Classify this French text:\n" + shot4_v2,},
            {'role': 'assistant', 'content': cot4_v2 + "\n" + "CECR Level: **" + classe2CECR[value4_v2] + "**"},
            {'role': 'user','content': "Classify this French text:\n" + text,},
            {'role': 'assistant', 'content': 'CECR Level: **'}
        ])
    elif prompt_type == "fr_CECR_few_shot_cot_v2": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    "Vous êtes un expert linguistique spécialisé dans l'évaluation des niveaux de français selon le Cadre européen commun de référence pour les langues (CECR). Votre tâche consiste à classer le texte français suivant dans l'un des niveaux du CECR : A1, A2, B1, B2, C1 ou C2.\n"
                    '\nExemple :'
                    'Texte à classifier : "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'Niveau CECR: **A1**'
                ),
            },
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot3_v2,},
            {'role': 'assistant', 'content': cot3_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value3_v2] + "**"},
            {'role': 'user','content': "Classifiez ce texte français :\n" + shot1_v2,},
            {'role': 'assistant', 'content': cot1_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value1_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot2_v2,},
            {'role': 'assistant', 'content': cot2_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value2_v2] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot4_v2,},
            {'role': 'assistant', 'content': cot4_v2 + "\n" + "Niveau CECR : **" + classe2CECR[value4_v2] + "**"},
            {'role': 'user','content': "Classifiez ce texte français :\n" + text,},
            {'role': 'assistant', 'content': 'Niveau CECR : **'}
        ])
    elif prompt_type == "fr_CECR_few_shot_cot_v3": # chain of thought
        response: ChatResponse = chat(model=model_name, messages=[
            {
                'role': 'system',
                'content': (
                    "Vous êtes un expert linguistique spécialisé dans l'évaluation des niveaux de français selon le Cadre européen commun de référence pour les langues (CECR). Votre tâche consiste à classer le texte français suivant dans l'un des niveaux du CECR : A1, A2, B1, B2, C1 ou C2.\n"
                    '\nExemple :'
                    'Texte à classifier : "Bonjour, je m\'appelle Jean. J\'habite à Paris. J\'aime jouer au football.'
                    'Le texte fourni est composé de phrases simples et courtes, utilisant des structures grammaticales de base et un vocabulaire élémentaire. Selon le Cadre européen commun de référence pour les langues (CECRL), le niveau A1 correspond à la capacité de comprendre et d\'utiliser des expressions familières et quotidiennes ainsi que des énoncés très simples visant à satisfaire des besoins concrets.'
                    'Niveau CECR: **A1**'
                ),
            },
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot3_v3,},
            {'role': 'assistant', 'content': cot3_v3 + "\n" + "Niveau CECR : **" + classe2CECR[value3_v3] + "**"},
            {'role': 'user','content': "Classifiez ce texte français :\n" + shot1_v3,},
            {'role': 'assistant', 'content': cot1_v3 + "\n" + "Niveau CECR : **" + classe2CECR[value1_v3] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot6_v3,},
            {'role': 'assistant', 'content': cot6_v3 + "\n" + "Niveau CECR : **" + classe2CECR[value6_v3] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot2_v3,},
            {'role': 'assistant', 'content': cot2_v3 + "\n" + "Niveau CECR : **" + classe2CECR[value2_v3] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot4_v3,},
            {'role': 'assistant', 'content': cot4_v3 + "\n" + "Niveau CECR : **" + classe2CECR[value4_v3] + "**"},
            {'role': 'user', 'content': "Classifiez ce texte français :\n" + shot5_v3,},
            {'role': 'assistant', 'content': cot5_v3 + "\n" + "Niveau CECR : **" + classe2CECR[value5_v3] + "**"},
            
            {'role': 'user','content': "Classifiez ce texte français :\n" + text,},
            {'role': 'assistant', 'content': 'Niveau CECR : **'}
        ])
    else:
        raise ValueError("Invalid prompt type. Must be 'en', 'fr', 'en_do_not', 'fr_do_not', 'fr_few_shot', 'fr_few_shot_cot', 'fr_few_shot_cot_with_protocol' or 'en_CECR'.")
    return response['message']['content']




def load_dataset(path="../../data/Qualtrics_Annotations_formatB.csv"):
    df = pd.read_csv(path)
    return df

def infer_classification(dataset, model_name, prompt_type, csv_path):
    # if file results/{prompt_type}.txt exists, load it
    if os.path.exists(f"results/llm_output/{model_name}_{prompt_type}.json"):
        with open(f"results/llm_output/{model_name}_{prompt_type}.json", encoding="utf-8") as f:
            text2output = json.load(f)  # Load the JSON file as a list of dictionaries [{"text_a": ..., "text_b": ...}, ...]
    else:
        text2output = dict()

    bar = progressbar.ProgressBar(maxval=len(dataset))
    i = 0
    for index, row in dataset.iterrows():
        if row["text"] in text2output:
            dataset.at[index, "difficulty"] = text2output[row["text"]]
        else:
            dataset.at[index, "difficulty"] = classify_text_difficulty(row["text"], model_name, prompt_type)
            text2output[row["text"]] = dataset.at[index, "difficulty"]
            with open(f"results/llm_output/{model_name}_{prompt_type}.json", "w", encoding="utf-8") as f:
                json.dump(text2output, f, ensure_ascii=False, indent=4)  # Pretty-print JSON
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


def evaluate_classification_old(dataset, confusion_matrix_path, results_path):
    pattern = r"(?:<|\*\*)(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe|A1|A2|B1|B2|C1|C2)(?:>|\*\*)"

    # Correction des valeurs erronées dans la colonne "difficulty"
    for index, row in dataset.iterrows():
        if row["difficulty"] not in ["Very Easy", "Easy", "Accessible", "Complex", "Très Facile", "Facile", "Accessible", "+Complexe", "A1", "A2", "B1", "B2", "C1", "C2"]:
            # print("Text:", row["text"])
            # print("Before:", row["difficulty"])

            matches = re.findall(pattern, row["difficulty"]) # Trouver toutes les occurrences
            if matches:
                predicted_class = matches[-1]  # Prendre la dernière occurrence
                if predicted_class in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                    predicted_class = CECR2classe[predicted_class]
                dataset.at[index, "difficulty"] = predicted_class
            else:
                matches = re.findall(r"(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe|A1|A2|B1|B2|C1|C2)", row["difficulty"])
                if matches:
                    predicted_class = matches[-1]
                    if predicted_class in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                        predicted_class = CECR2classe[predicted_class]
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







# Division en 5 parties
def split_into_folds(y_true, y_pred, n_splits=5):
    indices = np.array_split(np.arange(len(y_true)), n_splits)
    return [(y_true[idx], y_pred[idx]) for idx in indices]
def evaluate_classification(dataset, confusion_matrix_path, results_path):
    pattern = r"(?:<|\*\*)(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe|A1|A2|B1|B2|C1|C2)(?:>|\*\*)"

    # Correction des valeurs erronées dans la colonne "difficulty"
    for index, row in dataset.iterrows():
        if row["difficulty"] not in ["Very Easy", "Easy", "Accessible", "Complex", "Très Facile", "Facile", "Accessible", "+Complexe", "A1", "A2", "B1", "B2", "C1", "C2"]:
            # print("Text:", row["text"])
            # print("Before:", row["difficulty"])

            matches = re.findall(pattern, row["difficulty"]) # Trouver toutes les occurrences
            if matches:
                predicted_class = matches[-1]  # Prendre la dernière occurrence
                if predicted_class in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                    predicted_class = CECR2classe[predicted_class]
                dataset.at[index, "difficulty"] = predicted_class
            else:
                matches = re.findall(r"(Very Easy|Easy|Accessible|Complex|Très Facile|Facile|Accessible|\+Complexe|A1|A2|B1|B2|C1|C2)", row["difficulty"])
                if matches:
                    predicted_class = matches[-1]
                    if predicted_class in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                        predicted_class = CECR2classe[predicted_class]
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



    folds = split_into_folds(y_true, y_pred, n_splits=5)

    accuracies = []
    adjacent_accuracies = []
    macro_f1s = []

    for y_t, y_p in folds:
        # Calcul des métriques globales
        acc = accuracy_score(y_t, y_p)
        adj_acc = (abs(y_t - y_p) <= 1).mean()
        macro_f1 = f1_score(y_t, y_p, average='macro', zero_division=0)

        accuracies.append(acc)
        adjacent_accuracies.append(adj_acc)
        macro_f1s.append(macro_f1)

    print(accuracies)

    # Calcul des moyennes et écarts-types
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_adj_accuracy = np.mean(adjacent_accuracies)
    std_adj_accuracy = np.std(adjacent_accuracies)
    mean_macro_f1 = np.mean(macro_f1s)
    std_macro_f1 = np.std(macro_f1s)

    print(f"Global Accuracy: {mean_accuracy} ± {std_accuracy}")
    print(f"Global Adjacent Accuracy: {mean_adj_accuracy} ± {std_adj_accuracy}")
    print(f"Global Macro F1: {mean_macro_f1} ± {std_macro_f1}")

    txt = f"global_accuracy\t{mean_accuracy} ± {std_accuracy}\nglobal_adjacent_accuracy\t{mean_adj_accuracy} ± {std_adj_accuracy}\nglobal_macro_f1\t{mean_macro_f1} ± {std_macro_f1}\n"

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

'''
if __name__ == "__main__":
    model_name = "deepseek-r1:32b" # "deepseek-r1:70b" # "llama3.2:1b" # "deepseek-r1:70b" # "deepseek-r1:7b" # "llama3.2:1b"
    prompt_type = "fr_CECR_few_shot_cot_v2" # "en_CECR" # "en_CECR_few_shot_cot_v2" # "fr_CECR" # "fr_CECR_few_shot_cot_v3" # "en_CECR_few_shot_cot" # "fr_few_shot_cot_with_protocol" # "fr_few_shot_cot" # "fr_few_shot" # "fr_do_not" # "en_do_not" # "en" # "fr"
    dataset_path = "../../data/Qualtrics_Annotations_formatB.csv"
    csv_path = "./data/Qualtrics_Annotations_formatB_out_" + model_name + "_" + prompt_type + ".csv"
    confusion_matrix_path = "./results/confusion_matrix_" + model_name + "_" + prompt_type + ".png"
    results_path = "./results/results_" + model_name + "_" + prompt_type + ".txt"

    dataset = get_difficulty_level(dataset_path, model_name, prompt_type, csv_path) # infer or load the difficulty level

    print(dataset)
    # for each value of the column "difficulty", print value if not in ["Very Easy", "Easy", "Accessible", "Complex"]
    # print(dataset[~dataset["difficulty"].isin(["Very Easy", "Easy", "Accessible", "Complex"])]["difficulty"].unique())

    evaluate_classification(dataset, confusion_matrix_path, results_path) # evaluate the classification
'''

if __name__ == "__main__":
    model_name = "deepseek-r1:14b" # "deepseek-r1:7b" # "gemma3:27b" # "qwen2.5:72b" # "deepseek-r1:32b" # "deepseek-r1:70b" # "llama3.2:1b" # "deepseek-r1:70b" # "deepseek-r1:7b" # "llama3.2:1b"
    prompt_types = ["en_CECR", "fr_CECR", "fr_CECR_few_shot_cot_v2", "en_CECR_few_shot_cot_v2"] # "en_CECR" # "en_CECR_few_shot_cot_v2" # "fr_CECR" # "fr_CECR_few_shot_cot_v3" # "en_CECR_few_shot_cot" # "fr_few_shot_cot_with_protocol" # "fr_few_shot_cot" # "fr_few_shot" # "fr_do_not" # "en_do_not" # "en" # "fr"
    # prompt_types = ["en_CECR_few_shot_cot_v2"]
    dataset_path = "../../data/Qualtrics_Annotations_formatB.csv"


    for prompt_type in prompt_types:
        csv_path = "./data/Qualtrics_Annotations_formatB_out_" + model_name + "_" + prompt_type + ".csv"
        confusion_matrix_path = "./results/cm/confusion_matrix_" + model_name + "_" + prompt_type + ".png"
        results_path = "./results/results_" + model_name + "_" + prompt_type + ".txt"

        dataset = get_difficulty_level(dataset_path, model_name, prompt_type, csv_path) # infer or load the difficulty level

        print(dataset)
        # for each value of the column "difficulty", print value if not in ["Very Easy", "Easy", "Accessible", "Complex"]
        # print(dataset[~dataset["difficulty"].isin(["Very Easy", "Easy", "Accessible", "Complex"])]["difficulty"].unique())

        evaluate_classification(dataset, confusion_matrix_path, results_path) # evaluate the classification