import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import idx2numpy

def sigmoide(z):
    return 1 / (1 + np.exp(-z))
    
def softmax(l_z, k):
    #print(f"l_z={l_z} et k={k}")
    return np.exp(l_z[k]) / (np.sum(np.exp(l_z)))
    

def lire_alpha_digit(l_ind):
    fich = loadmat(current+"//data//binaryalphadigs.mat")
    rep = []
    
    for i in l_ind:
        lst_i = fich['dat'][i]
        rep.append([x.reshape(-1) for x in lst_i])
    
    return rep

def afficher_img(img, dim):
    # Prétraitement
    img = img.reshape(dim)
    # Affichage
    fig, ax = plt.subplots(facecolor='black') # mettre un fond noir pour mieux visualiser l'image binaire
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title('Image Binaire')
    plt.axis('off')
    plt.show()
    
def save_img(img, dim, file_name):
    # Prétraitement
    img = img.reshape(dim)
    # Affichage
    fig, ax = plt.subplots(facecolor='black') # mettre un fond noir pour mieux visualiser l'image binaire
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title('Image Binaire')
    plt.axis('off')
    plt.savefig(current+"//imgs//"+file_name)
    
def lire_mnist(file_name, is_X):
    # Charger le fichier IDX en tant qu'array NumPy
    data = idx2numpy.convert_from_file(current+"//data//"+file_name)
    
    if (is_X):
        # Binarisation des données
        threshold = 128
        data = np.where(data > threshold, 1, 0)
        
        data = np.array([x.reshape(-1) for x in data])
    
    else:
        # Création d'un vecteur one hot encoded
        create_array = lambda n: np.array([1 if i == n else 0 for i in range(10)])
        data = np.array([create_array(n) for n in data])
    
    return data

"""
data_bin = lire_mnist("train-images-idx3-ubyte", is_X=True)[:4,:]
afficher_img(data_bin[0], dim=(28,28))
afficher_img(data_bin[1], dim=(28,28))
afficher_img(data_bin[2], dim=(28,28))
afficher_img(data_bin[3], dim=(28,28))

data = idx2numpy.convert_from_file(current+"//data//"+"train-images-idx3-ubyte")[:4,:]
afficher_img(data[0], dim=(28,28))
afficher_img(data[1], dim=(28,28))
afficher_img(data[2], dim=(28,28))
afficher_img(data[3], dim=(28,28))
"""