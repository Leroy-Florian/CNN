import numpy as np

np.random.seed(1)  # la génération de l'aléatoire sera toujours la même a chaque exec

# var X et Y du graphique
# permet de visualiser l'evolution du reseau de neurone

import matplotlib.pyplot as plt

xGraphCostFunction = []
yGraphCostFunction = []


# fonction d'activation sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivé de la fonction d'activation
def sigmoid_prime(x):
    return x * (1 - x)


# ----- Creation des données
inputs = np.array([[0, 0, 0, 1],
                   [0, 0, 1, 1],
                   [0, 1, 1, 1],
                   [1, 0, 1, 1],
                   [1, 1, 1, 1]
                   ])

reponses = np.array([[0],
                    [0],
                    [1],
                    [0],
                    [1]
                    ])

# donnee d'inputs test; servira a entrainer le reseau
inputs_test = np.array([[1, 1, 1, 0],
                        [0, 1, 1, 0],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0]])

# -------------------------

# ----- Dimensions du reseau de neurones

nb_inputs_neurons = 4  # entree
nb_hidden_neurons = 4  # layer
nb_output_neurons = 1  # sortie

# -------------------------

# initialise les poids de maniere aleatoire en -1 et 1
hidden_layer_weights = 2 * np.random.random((nb_inputs_neurons, nb_hidden_neurons)) - 1
output_layer_weights = 2 * np.random.random((nb_hidden_neurons, nb_output_neurons)) - 1

# -------------------------
# Entrainement
# -------------------------

nb_training_iteration = 100000

for i in range(nb_training_iteration):
    # ------------------------- Feed Forward
    # propage les info dans le reseau de neurone

    input_layer = inputs
    hidden_layer = sigmoid(np.dot(input_layer, hidden_layer_weights))
    output_layer = sigmoid(np.dot(hidden_layer, output_layer_weights))

    # ------------------------- Back Propagation
    # Calcul du cout pour chaque donnée. Represente a quel point on est loin du résultat

    output_layer_error = (reponses - output_layer)
    print("erreur :" + str(output_layer_error))

    # calcul de la valeur avec laquel on va corriger le poids entre le hidden et l'output layer
    output_layer_delta = output_layer_error * sigmoid_prime(output_layer)

    # Quels sont les poids entre l'input layer et le hidden qui ont contribuer a l'erreur
    hidden_layer_error = np.dot(output_layer_delta, output_layer_weights.T)

    # calcul de la valeur avec laquelle on va corriger le poids entre l'input et l'hidden layer

    hidden_layer_delta = hidden_layer_error * sigmoid_prime(hidden_layer)

    # correction des poids

    output_layer_weights += np.dot(hidden_layer.T, output_layer_delta)
    hidden_layer_weights += np.dot(input_layer.T, hidden_layer_delta)

    # Affichage du couts

    if (i % 10) == 0:
        cout = str(np.mean((np.abs(output_layer_error))))  # calcul de la moyenne  de toute la valeur de nos erreur
        print("Cout : " + cout)

        # abscisse du graph -> iteration de la boucle d'apprentissage
        xGraphCostFunction.append(i)

        # ordonnee du graph -> valeur du cout
        v = float("{0:.3f}".format(float(cout)))
        yGraphCostFunction.append(v)

# -------------------------
# phase de test
# -------------------------

# propage nos info a traver le reseaux
# utilise un echantillion qu'il n'a pas encore vu

input_layer = inputs_test
hidden_layer = sigmoid(np.dot(input_layer, hidden_layer_weights))
output_layer = sigmoid(np.dot(hidden_layer, output_layer_weights))

# affiche le resultat
print("-------------")
print("Résultat : ")
print(str(output_layer))

# affiche le graphique
plt.plot(xGraphCostFunction, yGraphCostFunction)
plt.show()
