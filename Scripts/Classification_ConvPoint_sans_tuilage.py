# -*- coding: utf-8 -*-

print("""
--------------------------------------------------------------------------------------------------------------------
Rouler avec Python 3.6 (conda activate pytorch_convpoint_p36)

Licences nécessaires: aucune

Ce script permet de classifier un nuage de points avec un réseau ConvPoint entrainé

Intrants: 
    - fichier .las

Auteur: pbug
--------------------------------------------------------------------------------------------------------------------
""")

import collections
import datetime
import math
import os
import shutil
import time
import torch

torch.cuda.empty_cache()


def temps_ecoul(t1, t2):
    duree_sec_prec = round(t2 - t1, 3)
    duree_sec = int(round(t2 - t1))
    duree_min = int(round(duree_sec / 60))
    duree_hrs = int(math.floor(duree_min / 60))
    reste_min = duree_min - (duree_hrs * 60)
    if duree_sec < 1:
        return "{0} sec.".format(duree_sec_prec)
    elif duree_sec < 60:
        return "{0} sec.".format(duree_sec)
    elif duree_sec > 3600:
        return "{0} h. {1} min.".format(duree_hrs, reste_min)
    else:
        return "{0} min.".format(duree_min)


modeles_dispo = collections.OrderedDict()
modeles_dispo["1"] = ["DALES10ppm", "25", "8168", "state_dict_dales.pth", 4]
modeles_dispo["2"] = ["DALES50ppm", "10", "32672", "ConvPoint_nfnd_DALES50ppm911_32672pts_10blocksize_0drop_0p001lr_"
                                                   "20epochs_600iter_batch8_2workers_2021-09-18-11-38-46.pth", 4]
modeles_dispo["3"] = ["VexcelM28brut221", "20", "32672", "CP_fd_VexcelM28brut221_32672pts_20block_0drop_0p001lr_20ep_"
                                                         "600iter_batch8_2work_2021-12-16-13-37-31.pth", 5]
modeles_dispo["4"] = ["VexcelM27brut221Poteaux", "10", "32672", "CP_fd_VexcelM27brut221Poteaux_32672pts_10block_0drop_"
                                                         "0p001lr_20ep_600iter_batch8_2work_2022-05-19-14-35-41.pth", 6]
modeles_dispo["5"] = ["VexcelSherb331brut_5classes", "20", "32672", "CP_fd_Sherb331brut_5classes_32672pts_20block_"
                                                "0drop_0p001lr_20ep_600iter_batch8_2work_2023-02-09-17-54-07.pth", 7]
modeles_dispo["6"] = ["VexcelM28brut221_13ppmc", "20", "32672", "CP_fd_Sub3Nap13ppcm_3classes_32672pts_20block_0drop_"
                                                        "0p001lr_20ep_600iter_batch8_2work_2023-09-06-20-43-43.pth", 5]

print("")
systeme = input("Quel système est utilisé? (D pour Docker par défaut/W pour Windows/L pour Linux)\n").upper()
if systeme == "":
    systeme = "D"
if systeme == "D":
    chemin_CP = r"/opt/ogc/ConvPoint_Effigis"
elif systeme == "W":
    chemin_CP = r"C:\\Users\\pbug\\.conda\\envs\\pytorch_convpoint_p36\\ConvPoint"
else:
    chemin_CP = r"/home/ubuntu/anaconda2/envs/pytorch_convpoint_p36/ConvPoint"

if systeme == "D":
    doss_intrant = "."
else:
    print("")
    doss_intrant = input("Dans quel dossier se trouvent les fichiers las à classifier? \n")

maintenant = datetime.datetime.now().strftime('%Y%m%d_%Hh%M')
doss_trav = os.path.join(doss_intrant, "Classif_{0}".format(maintenant))
os.mkdir(doss_trav)

print("")
print("Quel modèle doit-on utiliser?")
for cle in modeles_dispo.keys():
    print("{0}: {1}".format(cle, modeles_dispo[cle][0]))
code_modele = input("")

if code_modele not in modeles_dispo.keys():
    sortie = input("Hey! Ce modèle n'est pas disponible.")
    exit()
else:
    blocksize0 = modeles_dispo[code_modele][1]
    nbpoints0 = modeles_dispo[code_modele][2]
    model_path = os.path.join(chemin_CP, 'models', modeles_dispo[code_modele][3])
    mode0 = modeles_dispo[code_modele][4]

print("")
batchsize0 = input("Quelle taille de batch veux-tu utiliser? (8 par défaut)\n")
if batchsize0 == "":
    batchsize0 = "8"

print("")
nbworkers0 = input("Quel nombre de workers veux-tu utiliser? (2 par défaut)\n")
if nbworkers0 == "":
    nbworkers0 = "2"


debut = time.time()


tuiles_las = [os.path.join(doss_intrant, fich1) for fich1 in os.listdir(doss_intrant) if fich1.upper().endswith(".LAS")]
nb_tuiles = len(tuiles_las)
print("")
print("{0} tuiles".format(nb_tuiles))
print("")

# Inférence
script_inference = os.path.join(chemin_CP, "examples/airborne_lidar/airborne_lidar_inference.py")
os.system("python {0} --num_workers {1} --batchsize {2} --modeldir {3} --rootdir {4} --npoints {5} --blocksize {6} "
          "--mode {7} --outdir {8}".format(script_inference, nbworkers0, batchsize0, model_path, doss_intrant,
                                           nbpoints0, blocksize0, mode0, doss_trav))

fin = time.time()
print("")
print("L'ensemble des traitements a pris {0}".format(temps_ecoul(debut, fin)))
