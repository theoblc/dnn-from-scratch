import numpy as np
import matplotlib.pyplot as plt
import utils
from principal_DBN_alpha import DBN
from principal_RBM_alpha import RBM
from principal_DNN_MNIST import DNN

def rbm_test(p, q, var, batch_size, lr, nb_epoch, alpha_digits, nb_iter_gibbs):
    # Création du RBM
    rbm_test = RBM(p, q, var)

    X = utils.lire_alpha_digit(alpha_digits)
    X = np.array([row for x in X for row in x])
    print(X.shape)

    # Entraînement du RBM
    rbm_test.train_RBM(X, batch_size, lr, nb_epoch)

    for _ in range(8):
        # Générer une image avec le RBM
        l_img = rbm_test.generer_image_RBM(nb_iter_gibbs=nb_iter_gibbs, nb_img=1)

        # Affichage de l'image avec Matplotlib
        utils.afficher_img(l_img[0], dim=(20, 16))
    

def dbn_test(tailles_couches, batch_size, lr, nb_epoch, alpha_digits, nb_iter_gibbs, number=None):
    # Création du DBN
    dbn_test = DBN(tailles_couches)

    #Initialisation des variables d'entraînements et des données
    X = utils.lire_alpha_digit(alpha_digits)
    X = np.array([row for x in X for row in x])
    print(X.shape)
    
    # Entraînement du DBN
    dbn_test.train_DBN(X, batch_size=batch_size, learning_rate=lr, nb_epoch=nb_epoch)
    
    if number != None :
        for i in range(8):

            # Générer une image avec le DBN
            l_img = dbn_test.generer_image_DBN(nb_iter_gibbs=nb_iter_gibbs, nb_img=1)

            # Affichage de l'image avec Matplotlib
            utils.save_img(l_img[0], dim=(20, 16), file_name="nbcar_"+str(number)+"_"+str(i+1))
    else:
        # Générer une image avec le DBN
        l_img = dbn_test.generer_image_DBN(nb_iter_gibbs=nb_iter_gibbs, nb_img=1)

        # Affichage de l'image avec Matplotlib
        utils.afficher_img(l_img[0], dim=(20, 16))
    
def dnn_test(architecture, batch_size, lr, epochs_RBM, epochs_retro, nb_train_data, pretrain):
    # Création du DNN
    dnn_test = DNN(architecture)
    
    # Initialisation des variables d'entraînements et des données
    X_train = utils.lire_mnist("train-images-idx3-ubyte", is_X=True)
    Y_train = utils.lire_mnist("train-labels-idx1-ubyte", is_X=False)
    X_test = utils.lire_mnist("t10k-images-idx3-ubyte", is_X=True)
    Y_test = utils.lire_mnist("t10k-labels-idx1-ubyte", is_X=False)
    
    # Diminution du dataset d'entraînement
    X_train = X_train[:nb_train_data,:]
    Y_train = Y_train[:nb_train_data]

    # Entraînement du DNN
    if (pretrain):
        print("-----pretraining-----")
        dnn_test.pretrain_DNN(X_train, batch_size=batch_size, learning_rate=lr, nb_epoch=epochs_RBM)
    print("-----training DNN-----")
    dnn_test.retropropagation(X_train, Y_train, batch_size=batch_size, learning_rate=lr, nb_epoch=epochs_retro)
    
    # Tester le DNN
    print("-----testing DNN-----")
    error_rate = dnn_test.test_DNN(X_test, Y_test)
    print("error_rate=", error_rate)
    
def dnn_test_bis(architecture, nb_train_data):
    batch_size = 5
    lr = 0.1
    epochs_RBM = 100
    epochs_retro = 200
    
    # Initialisation des variables d'entraînements et des données
    X_train = utils.lire_mnist("train-images-idx3-ubyte", is_X=True)
    Y_train = utils.lire_mnist("train-labels-idx1-ubyte", is_X=False)
    X_test = utils.lire_mnist("t10k-images-idx3-ubyte", is_X=True)
    Y_test = utils.lire_mnist("t10k-labels-idx1-ubyte", is_X=False)
    
    # Diminution du dataset d'entraînement
    X_train = X_train[:nb_train_data,:]
    Y_train = Y_train[:nb_train_data]
    
    print("-----training DNN classic-----")
    # Création du DNN
    dnn_test = DNN(architecture)
    dnn_test.retropropagation(X_train, Y_train, batch_size=batch_size, learning_rate=lr, nb_epoch=epochs_retro)
    error_classique = dnn_test.test_DNN(X_test, Y_test)

    # Pré-entraînement du DNN
    print("-----pretraining DNN-----")
    dnn_test_pretrain = DNN(architecture)
    dnn_test_pretrain.pretrain_DNN(X_train, batch_size=batch_size, learning_rate=lr, nb_epoch=epochs_RBM)
    dnn_test_pretrain.retropropagation(X_train, Y_train, batch_size=batch_size, learning_rate=lr, nb_epoch=epochs_retro)
    error_pretrain = dnn_test_pretrain.test_DNN(X_test, Y_test)
    # Tester le DNN
    #print("-----testing DNNs-----")
    
    return error_classique, error_pretrain

def fig_1():
    l_test = [i for i in range(2, 10)]
    pretrain_errors = []
    classiq_errors = []
    
    for nb_couches in l_test:
        print(f"Etape {nb_couches-1}/{len(l_test)}")
        
        # Créer l'architecture avec le bon nombre de couches
        l_archi = [784]
        for _ in range(nb_couches):
            l_archi.append(200)
        # Couche de classification obligatoire 
        l_archi.append(10)
        archi = tuple(l_archi)
        
        # Entraînement et tests des deux modèles (pré-entraîné et classique)
        classiq_err, pretrain_err = dnn_test_bis(architecture=archi, nb_train_data=10000)
        pretrain_errors.append(pretrain_err)
        classiq_errors.append(classiq_err)
        print("classiq_errors=", classiq_errors)
        print("pretrain_errors=", pretrain_errors)

    x_axis = l_test
    print("classiq_errors=", classiq_errors)
    print("pretrain_errors=", pretrain_errors)
    plt.plot(x_axis, pretrain_errors, color="red", label="DNN pré-entraîné")
    plt.plot(x_axis, classiq_errors, color="blue", label="DNN classique")
    plt.xlabel("Nombre de couches de 200 neurones")
    plt.ylabel("Taux d'erreur")
    plt.legend()
    plt.show()
    
def fig_2():
    l_test = [i*100 for i in range(1, 9)]
    pretrain_errors = []
    classiq_errors = []
    
    step = 1
    for nb_neurones in l_test:
        print(f"Etape {step}/{len(l_test)}")
        step+=1
        
        # Créer l'architecture avec le bon nombre de couches
        l_archi = [784, nb_neurones, nb_neurones, 10]
        archi = tuple(l_archi)
        
        # Entraînement et tests des deux modèles (pré-entraîné et classique)
        classiq_err, pretrain_err = dnn_test_bis(architecture=archi, nb_train_data=10000)
        pretrain_errors.append(pretrain_err)
        classiq_errors.append(classiq_err)
        print("classiq_errors=", classiq_errors)
        print("pretrain_errors=", pretrain_errors)

    x_axis = l_test
    print("classiq_errors=", classiq_errors)
    print("pretrain_errors=", pretrain_errors)
    plt.plot(x_axis, pretrain_errors, color="red", label="DNN pré-entraîné")
    plt.plot(x_axis, classiq_errors, color="blue", label="DNN classique")
    plt.xlabel("Nombre de neurones par couche (pour 2 couches)")
    plt.ylabel("Taux d'erreur")
    plt.legend()
    plt.show()
    
def fig_3():
    l_test = [1000, 3000, 7000, 10000, 30000, 60000]
    pretrain_errors = []
    classiq_errors = []
    
    step = 1
    for nb_data in l_test:
        print(f"Etape {step}/{len(l_test)}")
        step+=1
        
        # Créer l'architecture avec le bon nombre de couches
        l_archi = [784, 200, 200, 10]
        archi = tuple(l_archi)
        
        # Entraînement et tests des deux modèles (pré-entraîné et classique)
        classiq_err, pretrain_err = dnn_test_bis(architecture=archi, nb_train_data=nb_data)
        pretrain_errors.append(pretrain_err)
        classiq_errors.append(classiq_err)
        print("classiq_errors=", classiq_errors)
        print("pretrain_errors=", pretrain_errors)

    x_axis = l_test
    print("classiq_errors=", classiq_errors)
    print("pretrain_errors=", pretrain_errors)
    plt.plot(x_axis, pretrain_errors, color="red", label="DNN pré-entraîné")
    plt.plot(x_axis, classiq_errors, color="blue", label="DNN classique")
    plt.xlabel("Nombre de données d'entraînement")
    plt.ylabel("Taux d'erreur")
    plt.legend()
    plt.show()
        
"""
1 : [2]
2 : [2, 4]
3 : [2, 4, 11]
4 : [2, 4, 11, 35]
5 : [2, 4, 9, 11, 35]
6 : [2, 4, 9, 11, 25, 35]
7 : [2, 4, 9, 11, 25, 29, 35]
8 : [2, 4, 9, 11, 23, 25, 29, 35]
9 : [0, 2, 4, 9, 11, 23, 25, 29, 35]
10: [0, 2, 4, 9, 11, 20, 23, 25, 29, 35]
"""    
l_digits = [15]
rbm_test(p=320, q=500, var=0.1, batch_size=5, lr=0.1, nb_epoch=100, alpha_digits=l_digits, nb_iter_gibbs=50)

q = 400
couches = [(320, q), (q, q), (q, q), (q, q)]

dbn_test(tailles_couches=couches,
         batch_size=5, lr=0.1, nb_epoch=100, 
         alpha_digits=l_digits,
         nb_iter_gibbs=50, number=None)

print(q)
dnn_test(architecture=(784, q, q, q, q, 10), batch_size=5, lr=0.1, epochs_RBM=100, epochs_retro=200, nb_train_data=60000, pretrain=True)

# Lance la génération de la figure 1
fig_1()
# Lance la génération de la figure 2
fig_2()
# Lance la génération de la figure 3
fig_3()