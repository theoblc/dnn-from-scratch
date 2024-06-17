import numpy as np
import utils
    
class RBM:
    
    def __init__(self, p, q, variance):
        self.p = p
        self.q = q
        
        self.a = np.zeros(p)
        self.b = np.zeros(q)
        self.W = np.random.normal(loc=0, scale=np.sqrt(variance), size=(p, q))
        
    def entree_sortie(self, v): #taille n x p
        h = utils.sigmoide(self.b + np.dot(v, self.W))
        return h #taille n x q
    
    def sortie_entree(self, h): #taille n x q
        v = utils.sigmoide(self.a + np.dot(h, np.transpose(self.W)))
        return v #taille n x p
        
    def train_RBM(self, X, batch_size, learning_rate, nb_epoch):
        
        for epoch in range(nb_epoch):
            X_save = X.copy()
            np.random.shuffle(X)
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:min(i+batch_size, X.shape[0]),:]
                tb = X_batch.shape[0]
                #print("taille batch = ", tb)
                v0 = X_batch
                #print("v0=", v0)
                p_h_v0 = self.entree_sortie(v0)
                #print("p_h_v0=", p_h_v0)
                h0 = (np.random.rand(tb, self.q) < p_h_v0)*1
                #print("h0=", h0)
                p_v_h0 = self.sortie_entree(h0)
                #print("p_v_h0=", p_v_h0)
                v1 = (np.random.rand(tb, self.p) < p_v_h0)*1
                #print("v1=", v1)
                p_h_v1 = self.entree_sortie(v1)
                #print("p_h_v1=", p_h_v1)
                
                # Gradients
                grad_W = np.dot(np.transpose(v0), p_h_v0) - np.dot(np.transpose(v1), p_h_v1)
                grad_a = np.sum(v0 - v1, axis=0)
                grad_b = np.sum(p_h_v0 - p_h_v1, axis=0)
                #print(grad_W)
                #print(grad_a)
                #print(grad_b)
                
                # MAJ des poids et biais
                self.W += learning_rate*(grad_W/tb)
                self.a += learning_rate*(grad_a/tb)
                self.b += learning_rate*(grad_b/tb) 
                
            # Calcul de l'EQM pour chaque époque (CD-1)
            
            H_save = self.entree_sortie(X_save)
            #print("H_save=", H_save)
            X_gen = self.sortie_entree(H_save)
            #print("X_gen=", X_gen)
            #print(f"EQM à l'époque {epoch+1} :", round(np.mean((X_save - X_gen) ** 2), 5))
            """ """
        return self
                  
    def generer_image_RBM(self, nb_iter_gibbs, nb_img):
        l_img = []
        for i in range(nb_img):
            # Générer une image aléatoirement
            img = np.random.rand(1, 320)
            #print("img=", img)
            
            # Itérer avec l'échantillonneur de Gibbs
            for _ in range(nb_iter_gibbs):
                p_h_v0 = self.entree_sortie(img)
                h0 = (np.random.rand(1, self.q) < p_h_v0)*1
                p_v_h0 = self.sortie_entree(h0)
                img = (np.random.rand(1, self.p) < p_v_h0)*1
                #print("img=", img)
            
            print(f"image n°{i} :", img)
            l_img.append(img)
        return l_img