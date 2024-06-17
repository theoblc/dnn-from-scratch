import numpy as np
from principal_RBM_alpha import RBM
    
class DBN:
    
    def __init__(self, tailles_couches):
        self.nb_couches = len(tailles_couches)
        variance = 0.1
        # tailles_couches contiendra le nombre de couche
        # ainsi que les p et q de chacune des couches 
        self.l_rbm = [RBM(p, q, variance) for p,q in tailles_couches] # liste de RBM
    
    def train_DBN(self, X, batch_size, learning_rate, nb_epoch): 
        h = X.copy()
        count=0
        # On parcours la liste de RBM
        for rbm in self.l_rbm:
            print("couche n°", count)
            count+=1
            # Entraînement du RBM
            rbm.train_RBM(h, batch_size, learning_rate, nb_epoch)
            
            # La sortie devient l'entrée pour le RBM suivant
            h = rbm.entree_sortie(h)
        return self
                
    
    def generer_image_DBN(self, nb_iter_gibbs, nb_img):
        l_img = []
        for i in range(nb_img):
            
            # On considère le dernier rbm de la liste
            rbm = self.l_rbm[-1]
            # Générer une image aléatoirement
            img = np.random.rand(1, rbm.p)
            #print("img=", img)
            
            # Itérer avec l'échantillonneur de Gibbs
            for _ in range(nb_iter_gibbs):
                p_h_v0 = rbm.entree_sortie(img)
                h0 = (np.random.rand(1, rbm.q) < p_h_v0)*1
                p_v_h0 = rbm.sortie_entree(h0)
                img = (np.random.rand(1, rbm.p) < p_v_h0)*1
            
            # On retire le dernier RBM
            lst_rbm = self.l_rbm[:-1]
            # On parcours la liste des RBM à l'envers pour le sens descendant
            for rbm in lst_rbm[::-1]:
                p_v_h = rbm.sortie_entree(img)
                img = (np.random.rand(1, rbm.p) < p_v_h)*1
                print(f"rbm (p={rbm.p}, q={rbm.q}), img={img}")
                
            print(f"image n°{i} :", img)
            l_img.append(img)
        return l_img
