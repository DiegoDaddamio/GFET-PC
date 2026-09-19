import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import concurrent.futures
import multiprocessing

class grphn():
    def __init__(self):
        self.h = 6.626*1e-34
        hbar = self.h/(2*np.pi)
        self.hbarr = 1

        self.q = 1.602176620e-19 # C

        self.k_B = 1.380649e-23 # J/T
        self.k_Bev = self.k_B/self.q

        self.d = 1.42 # A°
        self.a= np.sqrt(3)*self.d
        self.b = 4*np.pi/(np.sqrt(3)*self.a) # 1/A°

        self.a_1 = np.array([np.sqrt(3)/2,-1/2])*self.a
        self.a_2 = np.array([np.sqrt(3)/2,+1/2])*self.a
        self.b_1 = np.array([1/np.sqrt(3),-1])*(2*np.pi/self.a)
        self.b_2 = np.array([1/np.sqrt(3),+1])*(2*np.pi/self.a)

        self.S = self.a**2 * np.sqrt(3)/2

        self.S_r = ((2*np.pi)**2)/self.S

        self.t = 2.8 # eV
        self.v_F = self.t*np.sqrt(3)*self.a/(2*hbar)
        self.v_F_hbr = self.t*np.sqrt(3)*self.a/2

        self.N_recip = 100

        self.V_Dirac = 0


        self.eps_0 = 8.854e-14 #F/cm
        self.eps_ox = 3.9# SiO2
        self.d_ox = 300e-7 
        self.C_ox = (self.eps_0 * self.eps_ox) / self.d_ox  # F/cm^2

    def fFD(self,E,Ef,T): # Peut-être mettre tanh
        f = 1+np.exp((E-Ef)/(self.k_Bev*T))
        return 1/f
    
    def set_kpoint(self,kx,ky):
        self.kxx = kx
        self.kyy = ky

    def set_recip_grid(self,kx,ky):
        self.kx = kx
        self.ky = ky
        self.kxx, self.kyy = np.meshgrid(self.kx,self.ky)

    def set_Fermilvl(self,Ef):
        self.Fermilvl = Ef

    def sq_recip_grid(self):

        self.kx = np.linspace(-self.b/np.sqrt(3),self.b/np.sqrt(3),self.N_recip)
        self.ky = np.linspace(-self.b/np.sqrt(3),self.b/np.sqrt(3),self.N_recip)
        self.kxx, self.kyy = np.meshgrid(self.kx,self.ky)

    def set_FBZ(self):

        self.sq_recip_grid()

        radius_k = self.b / np.sqrt(3)

        mask = (np.abs(self.kxx) <= radius_k * np.sqrt(3) / 2) & \
            (np.abs(np.sqrt(3) * self.kyy + self.kxx) <= radius_k * np.sqrt(3)) & \
            (np.abs(np.sqrt(3) * self.kyy - self.kxx) <= radius_k * np.sqrt(3))
        
        self.n_k_FBZ = np.sum(mask)

        self.kxx = np.where(mask, self.kxx, np.nan)
        self.kyy = np.where(mask, self.kyy, np.nan)

    def get_recip_grid(self):
        return self.kxx, self.kyy

    def f(self):
        e1 = np.exp(-1j*(self.kxx*self.a_1[0]+self.kyy*self.a_1[1]))
        e2 = np.exp(-1j*(self.kxx*self.a_2[0]+self.kyy*self.a_2[1]))
        return 1 + e1 + e2

    def E_Tb(self,epsi=0):
        CB = epsi + np.abs(self.t)*np.abs(self.f())
        VB = epsi - np.abs(self.t)*np.abs(self.f())
        return VB, CB
    
    def Dirac_c(self,qxx,qyy,epsi=0):
        """Calcul autour d'un point K/K'"""
        CB = epsi + self.hbarr*self.v_F_hbr*np.sqrt(qxx**2 +qyy**2)
        VB = epsi - self.hbarr*self.v_F_hbr*np.sqrt(qxx**2 +qyy**2)

        return VB, CB
    
    def DOS_Dirac_ana(self, E_array):
        """
        DOS strictement analytique du cône de Dirac 
        (Formule : 2*S/(pi * (hv_F)^2) * |E|)
        """
        hv_F = self.hbarr * self.v_F_hbr
        return (np.abs(E_array) * 2 * self.S) / (np.pi * hv_F**2)

    def DOS(self, n_points=7500, n_bins=3000, E_max=None):
        """
        Calcule la DOS par histogramme sur la FBZ.
        n_points : résolution de la grille k
        n_bins   : nombre de bins en énergie
        E_max    : énergie max (défaut: 3*t)
        """
        if E_max is None:
            E_max = 3 * self.t  # max physique de |f|=3 → E=3t

        # Grille k temporaire (sans écraser self.kxx/kyy)
        kx = np.linspace(-self.b, self.b, n_points)
        ky = np.linspace(-self.b, self.b, n_points)
        kxx, kyy = np.meshgrid(kx, ky)

        # Masque FBZ
        radius_k = self.b / np.sqrt(3)
        mask = (np.abs(kxx) <= radius_k * np.sqrt(3) / 2) & \
            (np.abs(np.sqrt(3) * kyy + kxx) <= radius_k * np.sqrt(3)) & \
            (np.abs(np.sqrt(3) * kyy - kxx) <= radius_k * np.sqrt(3))
        
        self.n_k_FBZ = np.sum(mask)

        # Sauvegarde temporaire et calcul
        kxx_bak, kyy_bak = self.kxx, self.kyy
        self.kxx, self.kyy = kxx, kyy
        _, CB = self.E_Tb()
        self.kxx, self.kyy = kxx_bak, kyy_bak  # restauration

        # Histogramme
        CB_masked = CB[mask]
        E_levels = np.linspace(0, E_max, n_bins + 1)
        dE = E_levels[1] - E_levels[0]
        DOSp, _ = np.histogram(CB_masked, bins=E_levels)

        self.DOSp= DOSp.copy()

        # Normalisation : facteur 2 pour spin
        norm_factor = 2 / (len(CB_masked) * dE)
        DOSp = DOSp * norm_factor

        # Symétrie électron-trou
        DOSm = DOSp[::-1]
        self.DOS = np.concatenate([DOSm, DOSp])
        E_centers_p = (E_levels[:-1] + E_levels[1:]) / 2
        self.E_DOS = np.concatenate([-E_centers_p[::-1], E_centers_p])

        self.E_DOSp = E_centers_p.copy()

        return self.E_DOS, self.DOS, DOSp
    
    def DOS_chunked(self, n_points=15000, n_bins=3000, E_max=None):
            if E_max is None:
                E_max = 3 * self.t

            kx_array = np.linspace(-self.b, self.b, n_points)
            ky_array = np.linspace(-self.b, self.b, n_points)
            
            radius_k = self.b / np.sqrt(3)
            
            # Ce tableau va accumuler les résultats sans jamais stocker la grille 2D
            E_levels = np.linspace(0, E_max, n_bins + 1)
            DOSp_accumulated = np.zeros(n_bins)
            
            n_k_FBZ_total = 0 # Compteur 

            # On sauvegarde les kxx, kyy de la classe
            kxx_bak, kyy_bak = getattr(self, 'kxx', None), getattr(self, 'kyy', None)

            # /!\ CHUNKING: On traite la grille kx par blocs de lignes (ex: 500 par 500)
            # pour aller vite en Python tout en gardant une RAM très faible (< 100 Mo)
            chunk_size = 500 
            
            for i in range(0, n_points, chunk_size):
                # Extraction d'un bloc de kx
                kx_chunk = kx_array[i : i+chunk_size]
                
                # Création du "petit" meshgrid local
                kxx, kyy = np.meshgrid(kx_chunk, ky_array)
                
                # Masque local
                mask = (np.abs(kxx) <= radius_k * np.sqrt(3) / 2) & \
                    (np.abs(np.sqrt(3) * kyy + kxx) <= radius_k * np.sqrt(3)) & \
                    (np.abs(np.sqrt(3) * kyy - kxx) <= radius_k * np.sqrt(3))
                
                n_k_FBZ_total += np.sum(mask)
                
                # Calcul local de l'énergie
                self.kxx, self.kyy = kxx, kyy
                _, CB = self.E_Tb()
                
                # Histogramme local cumulatif
                hist, _ = np.histogram(CB[mask], bins=E_levels)
                DOSp_accumulated += hist

            # Restauration
            self.kxx, self.kyy = kxx_bak, kyy_bak
            self.n_k_FBZ = n_k_FBZ_total

            # --- Normalisation (Identique à avant) ---
            dE = E_levels[1] - E_levels[0]
            dk_x = (2*self.b) / n_points
            dk_y = (2*self.b) / n_points
            area_per_point = dk_x * dk_y
            
            norm_factor = 2 / (self.n_k_FBZ * dE) 
            DOSp = DOSp_accumulated * norm_factor
            
            # Symétrie
            DOSm = DOSp[::-1]
            self.DOS = np.concatenate([DOSm, DOSp])
            E_centers_p = (E_levels[:-1] + E_levels[1:]) / 2
            self.E_DOS = np.concatenate([-E_centers_p[::-1], E_centers_p])
            
            return self.E_DOS, self.DOS, DOSp


    def C_q(self, T, Fermilvl):
        kT_eV = self.k_Bev * T 
        
        # Variable réduite x = (E - Ef) / kT
        x = (self.E_DOS - Fermilvl) / kT_eV
        
        f_p = np.where(np.abs(x) < 80, 1 / (4 * kT_eV * np.cosh(x / 2)**2), 0.0)
        
        integral_cv = np.trapezoid(self.DOS * f_p, self.E_DOS)
        
        cq_uF = integral_cv * self.q * (1e22 / self.S)
        
        return cq_uF # en uF/cm^2
    
    def get_n(self, Ef, T):
        return np.trapezoid(self.DOS * self.fFD(self.E_DOS, Ef, T), self.E_DOS)
        
    def get_net_density(self, Ef, T):

        c_ref = self.get_n(0, T)
        c_ml = self.get_n(Ef, T)
        n_per_cell = c_ml - c_ref
        
        n_cm2 = n_per_cell * (1e16 / self.S)
        return n_cm2

    def dep_Voltage(self, T, C_ox, V_dirac=0,Ef_inf=-1,Ef_sup=1,N=300):

        Ef_array = np.linspace(Ef_inf, Ef_sup, N) 
        
        Vg_array = np.zeros_like(Ef_array)
        n_array = np.zeros_like(Ef_array)
        
        for i, ef in enumerate(Ef_array):

            n_net = self.get_net_density(ef, T)
            n_array[i] = n_net
            
            Vg = V_dirac + (self.q * n_net / C_ox) + ef
            
            Vg_array[i] = Vg
            
        return Vg_array, n_array, Ef_array
    
    def DOS_parallel(self, n_points=15000, n_bins=3000, E_max=None, chunk_size=200):
        """
        Calcul de DOS hautement optimisé (Parallélisé & RAM-friendly)
        """
        if E_max is None:
            E_max = 3 * self.t

        kx_array = np.linspace(-self.b, self.b, n_points)
        ky_array = np.linspace(-self.b, self.b, n_points)
        
        E_levels = np.linspace(0, E_max, n_bins + 1)
        DOSp_accumulated = np.zeros(n_bins)
        
        n_k_FBZ_total = 0 

        # Diviser l'array kx en petits morceaux
        chunks = [kx_array[i : i+chunk_size] for i in range(0, n_points, chunk_size)]
        
        # On utilise tous les cœurs du processeur disponibles
        n_cores = multiprocessing.cpu_count()
        print(f"Calcul de de la DOS sur {n_cores} cœurs (n_points={n_points})...")

        # Lancement de la meute de processus en parallèle
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            # On soumet chaque sous-bloc à un processus
            futures = [executor.submit(_compute_chunk, chk, ky_array, self.b, self.a_1, self.a_2, self.t, E_levels) for chk in chunks]
            
            # On récolte les résultats au fur et à mesure qu'ils se terminent
            for future in concurrent.futures.as_completed(futures):
                hist, n_k = future.result()
                DOSp_accumulated += hist
                n_k_FBZ_total += n_k

        self.n_k_FBZ = n_k_FBZ_total

        # --- Normalisation (Identique) ---
        dE = E_levels[1] - E_levels[0]
        norm_factor = 2 / (self.n_k_FBZ * dE) 
        DOSp = DOSp_accumulated * norm_factor
        
        # Symétrie
        DOSm = DOSp[::-1]
        self.DOS = np.concatenate([DOSm, DOSp])
        
        E_centers_p = (E_levels[:-1] + E_levels[1:]) / 2
        self.E_DOS = np.concatenate([-E_centers_p[::-1], E_centers_p])
        self.E_DOSp = E_centers_p.copy()
        
        print("Fin du calcul !")
        
        return self.E_DOS, self.DOS, DOSp
    
def _compute_chunk(kx_chunk, ky_array, b, a_1, a_2, t, E_levels):
    """
    Fonction isolée qui calcule l'histogramme pour un sous-bloc.
    Elle est placée hors de la classe pour un parallélisme ultra-léger en RAM.
    """
    kxx, kyy = np.meshgrid(kx_chunk, ky_array)

    # Masque FBZ
    radius_k = b / np.sqrt(3)
    mask = (np.abs(kxx) <= radius_k * np.sqrt(3) / 2) & \
        (np.abs(np.sqrt(3) * kyy + kxx) <= radius_k * np.sqrt(3)) & \
        (np.abs(np.sqrt(3) * kyy - kxx) <= radius_k * np.sqrt(3))

    # ---- L'ASTUCE RAM ----
    # On extrait uniquement les points valides, ce qui détruit le tableau 2D super lourd 
    # et crée un vecteur 1D 40% plus petit !
    kxx_m = kxx[mask]
    kyy_m = kyy[mask]
    
    n_k_chunk = len(kxx_m)

    # Calcul Tight-Binding uniquement sur la FBZ
    e1 = np.exp(-1j * (kxx_m * a_1[0] + kyy_m * a_1[1]))
    e2 = np.exp(-1j * (kxx_m * a_2[0] + kyy_m * a_2[1]))
    CB_masked = np.abs(t) * np.abs(1 + e1 + e2)

    # Histogramme local cumulatif
    hist, _ = np.histogram(CB_masked, bins=E_levels)
    
    # On renvoie juste le petit array de l'histogramme, ça pèse 0 en RAM !
    return hist, n_k_chunk