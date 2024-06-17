import numpy as np
import sys
import utils
from principal_DBN_alpha import DBN
from principal_RBM_alpha import RBM

class DNN:
    
    def __init__(self, architecture):
        # Il faut décomposer l'architecture (n, m, p) -> [(n, m), (m, p)] 
        # sans oublier la couche de classification (dernière couche)
        taille_arch = len(architecture)
        tailles_couches = [(architecture[i],architecture[i+1]) for i in range(taille_arch-2)]
        p, q = architecture[taille_arch-2], architecture[taille_arch-1]
        var = 0.1
        
        self.nb_couches = taille_arch
        self.dbn = DBN(tailles_couches)
        self.classif_couche = RBM(p, q, var)
        
        self.trained = False
        
    def pretrain_DNN(self, X, batch_size, learning_rate, nb_epoch):
        self.dbn = self.dbn.train_DBN(X, batch_size, learning_rate, nb_epoch)
        return self
    
    def calcul_softmax(self, X):
        W = self.classif_couche.W
        b = self.classif_couche.b
        x = np.dot(X, W) + b
        
        if (len(x.shape)==1):
            lst_probas = np.exp(x) / np.sum(np.exp(x))
        else:
            lst_probas = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        #print("lst_probas=", lst_probas)
        
        return lst_probas
    
    def entree_sortie_reseau(self, X):
        l_couches_sorties = []
        
        # Couches du DBN
        h = X
        l_couches_sorties.append(h)
        for rbm in self.dbn.l_rbm:
            h = rbm.entree_sortie(h)
            l_couches_sorties.append(h)
        
        # Couche de classification
        l_probas = self.calcul_softmax(h)
        #h = np.argmax(l_probas, axis=1) # argmax par ligne !!!! et pas sur tout le tableau
        l_couches_sorties.append(l_probas)
        
        return l_couches_sorties 
    
    def retropropagation(self, X, Y, batch_size, learning_rate, nb_epoch):
        for epoch in range(nb_epoch):
            X_save = X.copy()
            Y_save = Y.copy()
            
            # Appliquez une même permutation aléatoire à X et Y
            indices_permutation = np.random.permutation(len(X))
            X = X[indices_permutation]
            Y = Y[indices_permutation]
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:min(i+batch_size, X.shape[0]),:]
                Y_batch = Y[i:min(i+batch_size, Y.shape[0])]
                
                tb = X_batch.shape[0]
                
                l_couches_sorties = self.entree_sortie_reseau(X_batch)
                # La prédiction Y_hat correspont à la sortie de la dernière couche (classification)
                Y_hat = l_couches_sorties[-1]
                #print("Y_hat=", Y_hat)
                #print(Y_hat.shape)
                #print("Y_batch=",Y_batch)
                #print(Y_batch.shape)
                
                c_l = Y_hat - Y_batch
                # Gradients
                grad_W = np.dot(np.transpose(c_l), l_couches_sorties[len(l_couches_sorties)-2])
                grad_b = np.sum(c_l, axis=0)
                #print("grad_W=", grad_W, grad_W.shape)
                #print("grad_b=", grad_b, grad_b.shape)

                # Sauvegarde avant MAJ de W
                W_save = self.classif_couche.W
                # MAJ des poids et biais
                self.classif_couche.W -= learning_rate*(np.transpose(grad_W)/tb)
                self.classif_couche.b -= learning_rate*(grad_b/tb)
                
                for l in range(self.nb_couches-1, 1, -1):
                    x_l = l_couches_sorties[l-1]
                    #print("x_l=", x_l, x_l.shape)
                    c_l = np.dot(c_l, np.transpose(W_save))*x_l*(1-x_l)
                    #print("c_l=", c_l, c_l.shape)
                    
                    # Gradients
                    grad_W = np.dot(np.transpose(c_l), l_couches_sorties[l-2])
                    grad_b = np.sum(c_l, axis=0)
                
                    # Sauvegarde avant MAJ de W
                    W_save = self.dbn.l_rbm[l-2].W
                    # MAJ des poids et biais
                    self.dbn.l_rbm[l-2].W -= learning_rate*(np.transpose(grad_W)/tb)
                    self.dbn.l_rbm[l-2].b -= learning_rate*(grad_b/tb) 
                
            l_couches_sorties = self.entree_sortie_reseau(X_save)
            Y_hat = l_couches_sorties[-1]
            #print("Y_hat=", Y_hat.shape)
            #print("Y_save=", Y_save.shape)
            #cross_entropy = - np.dot(Y_save, np.transpose(np.log(Y_hat)))
            cross_entropy = np.mean([- np.dot(Y_save[i,:], np.transpose(np.log(Y_hat[i,:]))) for i in range(Y_save.shape[0])])
            print(f"Cross-Entropy à l'époque {epoch+1} :", cross_entropy)
            
        self.trained = True
        return self
    
    def test_DNN(self, X_test, Y_test):
        if not(self.trained):
            print("Le modèle n'a pas été entraîné.", file=sys.stderr)
            return
        else:
            l_couches_sorties = self.entree_sortie_reseau(X_test)
            Y_hat = l_couches_sorties[-1]
            
            # Cross-Entropy
            cross_entropy = np.mean([- np.dot(Y_test[i,:], np.transpose(np.log(Y_hat[i,:]))) for i in range(Y_test.shape[0])])
            print("Cross-Entropy :", cross_entropy)
            
            # Taux d'erreur
            #print("probas_sortie=", Y_hat)
            Y_hat_0_1 = np.argmax(Y_hat, axis=1)
            Y_test = np.argmax(Y_test, axis=1)
            #print("true_label=", Y_test)
            
            errors = np.sum(Y_test != Y_hat_0_1)
            error_rate = errors / len(Y_test)
            print("Taux d'erreur :", error_rate)
            
            return error_rate