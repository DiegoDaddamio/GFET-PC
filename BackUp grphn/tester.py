import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import scipy.special as sp

# =========================================================================
# 1. LA FONCTION TRAVAILLEUR (En dehors de la classe)
# =========================================================================
def _compute_chunk(kx_chunk, ky_array, b, a_1, a_2, t, E_levels):
    """Calcule l'histogramme pour un sous-bloc de la grille kx."""
    kxx, kyy = np.meshgrid(kx_chunk, ky_array)

    # Masque FBZ
    radius_k = b / np.sqrt(3)
    mask = (np.abs(kxx) <= radius_k * np.sqrt(3) / 2) & \
           (np.abs(np.sqrt(3) * kyy + kxx) <= radius_k * np.sqrt(3)) & \
           (np.abs(np.sqrt(3) * kyy - kxx) <= radius_k * np.sqrt(3))

    # Astuce RAM : On extrait uniquement les points valides (vecteurs 1D)
    kxx_m = kxx[mask]
    kyy_m = kyy[mask]
    n_k_chunk = len(kxx_m)

    # Calcul TB sur les bons points uniquement
    e1 = np.exp(-1j * (kxx_m * a_1[0] + kyy_m * a_1[1]))
    e2 = np.exp(-1j * (kxx_m * a_2[0] + kyy_m * a_2[1]))
    CB_masked = np.abs(t) * np.abs(1 + e1 + e2)

    # Histogramme
    hist, _ = np.histogram(CB_masked, bins=E_levels)
    
    return hist, n_k_chunk

# =========================================================================
# 2. DEFINITION DE TA CLASSE (Résumé)
# =========================================================================
class grphn():
    def __init__(self):
        # Constantes de base
        self.h = 6.626e-34
        hbar = self.h / (2*np.pi)
        self.hbarr = 1
        self.t = 2.7 # eV
        self.d = 1.42 # A
        self.a = np.sqrt(3) * self.d
        self.b = 4*np.pi / (np.sqrt(3)*self.a)
        
        # Vecteurs
        self.a_1 = np.array([np.sqrt(3)/2, -1/2]) * self.a
        self.a_2 = np.array([np.sqrt(3)/2, 1/2]) * self.a
        self.b_1 = np.array([1/np.sqrt(3), -1]) * (2*np.pi/self.a)
        self.b_2 = np.array([1/np.sqrt(3), 1]) * (2*np.pi/self.a)
        
        self.S = self.a**2 * np.sqrt(3) / 2
        self.v_F = self.t * np.sqrt(3) * self.a / (2*hbar)
        self.v_F_hbr = self.t * np.sqrt(3) * self.a / 2

    def DOS_Dirac_ana(self, E_array):
        # Formule analytique exacte
        hv_F = self.hbarr * self.v_F_hbr
        return (np.abs(E_array) * 2 * self.S) / (np.pi * hv_F**2)

    def DOS_parallel(self, n_points=5000, n_bins=3000, E_max=None, chunk_size=200):
        if E_max is None:
            E_max = 3 * self.t

        kx_array = np.linspace(-self.b, self.b, n_points)
        ky_array = np.linspace(-self.b, self.b, n_points)
        
        E_levels = np.linspace(0, E_max, n_bins + 1)
        DOSp_accumulated = np.zeros(n_bins)
        
        n_k_FBZ_total = 0 
        chunks = [kx_array[i : i+chunk_size] for i in range(0, n_points, chunk_size)]
        
        n_cores = multiprocessing.cpu_count()
        print(f"-> Début calcul DOS sur {n_cores} cœurs (n_points={n_points})...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [executor.submit(_compute_chunk, chk, ky_array, self.b, self.a_1, self.a_2, self.t, E_levels) for chk in chunks]
            
            for future in concurrent.futures.as_completed(futures):
                hist, n_k = future.result()
                DOSp_accumulated += hist
                n_k_FBZ_total += n_k

        self.n_k_FBZ = n_k_FBZ_total

        dE = E_levels[1] - E_levels[0]
        norm_factor = 2 / (self.n_k_FBZ * dE) 
        DOSp = DOSp_accumulated * norm_factor
        
        DOSm = DOSp[::-1]
        self.DOS = np.concatenate([DOSm, DOSp])
        
        E_centers_p = (E_levels[:-1] + E_levels[1:]) / 2
        self.E_DOS = np.concatenate([-E_centers_p[::-1], E_centers_p])
        
        print("-> Fin du calcul !")
        return self.E_DOS, self.DOS, DOSp

# =========================================================================
# 3. LANCEMENT DU SCRIPT (Le fameux "if __name__ == '__main__'")
# =========================================================================
if __name__ == '__main__':
    
    # 1. Instanciation
    graphene = grphn()
    
    # 2. Lancement du calcul super rapide
    # Test avec 6000 points (ça équivaut à une grille de 36 millions de points !)
    graphene.DOS_parallel(n_points=30000, n_bins=500,E_max=2)
    
    # 3. Récupération pour l'affichage (côté énergie positive)
    mask = graphene.E_DOS >= 0
    E_pos = graphene.E_DOS[mask]
    DOS_pos = graphene.DOS[mask]
    DOS_ana = graphene.DOS_Dirac_ana(E_pos)

    er = np.abs(DOS_pos-DOS_ana)/np.abs(DOS_pos) *100
    
    # 4. Tracé du graphe de vérification
    plt.figure(figsize=(8, 5))
    
    # plt.plot(E_pos, DOS_pos, color='#464c89', lw=2, label="Tight-Binding (Multicoeur)")
    # plt.plot(E_pos, DOS_ana, color='#ff6b59', lw=2, linestyle='--', label="Cône de Dirac (Analytique)")
    
    plt.semilogy(E_pos,er, color='#464c89', lw=2)

    # On marque la singularité de Van Hove
    # plt.axvline(x=2.7, color='grey', linestyle=':', label='Singularité (E = t = 2.7 eV)')
    
    plt.xlabel(r"Énergie $E$ (eV)", fontsize=12)
    plt.ylabel(r"DOS (états / eV / maille)", fontsize=12)
    plt.title("Densité d'états du Graphène : TB vs Dirac", fontsize=14)
    # plt.xlim(0, 3 * 2.7)
    # plt.ylim(0, np.max(DOS_pos) * 1.1)
    
    # plt.legend(framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()