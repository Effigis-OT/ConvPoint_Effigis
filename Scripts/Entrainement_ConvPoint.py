# -*- coding: utf-8 -*-

print("""
--------------------------------------------------------------------------------------------------------------------
Rouler avec Python 3.6 (pytorch_convpoint_p36)

Licences nécessaires: aucune

Ce script permet d'entrainer un réseau ConvPoint

Intrants: 
    - fichiers .las annotés

Auteur: pbug
--------------------------------------------------------------------------------------------------------------------
""")

import collections
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

print("")
modele_pth = input("Quel est le chemin du modèle pth en entrée? (Enter pour DALES50ppm)\n")
if modele_pth == "":
    modele_pth = os.path.join(chemin_CP, "models/ConvPoint_nfnd_DALES50ppm911_32672pts_10blocksize_0drop_0p001lr_"
                                         "20epochs_600iter_batch8_2workers_2021-09-18-11-38-46.pth")

print("")
blocksize0 = input("Quel taille de bloc veux-tu utiliser? (20 par défaut)\n")
if blocksize0 == "":
    blocksize0 = "20"

print("")
nbpoints0 = input("Quel nombre de points veux-tu utiliser? (32672 par défaut)\n")
if nbpoints0 == "":
    nbpoints0 = "32672"

print("")
iter0 = input("Quel nombre d'itérations veux-tu utiliser? (600 par défaut)\n")
if iter0 == "":
    iter0 = "600"

print("")
batchsize0 = input("Quelle taille de batch veux-tu utiliser? (8 par défaut)\n")
if batchsize0 == "":
    batchsize0 = "8"

print("")
nbworkers0 = input("Quel nombre de workers veux-tu utiliser? (2 par défaut)\n")
if nbworkers0 == "":
    nbworkers0 = "2"

print("")
nbepochs0 = input("Quel nombre d'epochs veux-tu utiliser? (20 par défaut)\n")
if nbepochs0 == "":
    nbepochs0 = "20"

print("")
lr0 = input("Quel learning rate veux-tu utiliser? (1e-3 par défaut)\n")
if lr0 == "":
    lr0 = 1e-3
else:
    lr0 = float(lr0)

print("")
option_finetune = input("Veux-tu faire du finetuning? (O par défaut/N)\n").upper()
if option_finetune == "O" or option_finetune == "":
    option_finetune = "YES"
else:
    option_finetune = "NO"

modes_dispo = collections.OrderedDict()
modes_dispo["1"] = "3 classes: building, water, ground."
modes_dispo["2"] = "5 classes: building, water, ground, low vegetation and medium + high vegetation"
modes_dispo["3"] = "6 classes: building, water, ground, low vegetation, medium and high vegetation"
modes_dispo["4"] = "8 classes: ground, vegetation, cars, trucks, power lines, fences, poles and buildings"
modes_dispo["5"] = "3 classes: ground, vegetation, building"
modes_dispo["6"] = "4 classes: ground, vegetation, building, pole"
modes_dispo["7"] = "5 classes: ground, vegetation, building, power line, pole"

print("")
print("Quelles sont les classes requises?")
for cle in modes_dispo.keys():
    print("{0}: {1}".format(cle, modes_dispo[cle]))
code_classes = input("")
if code_classes == "4":
    option_decapit = "NO"
else:
    option_decapit = "YES"

print("")
nom_entrain = input("Quel est le nom du dataset d'entrainement? \n")

print("")
separation_auto = input("Les fichiers las d'entrainement doivent-ils être répartis automatiquement entre trn, val "
                        "et tst? (O/N par défaut) \n").upper()
if separation_auto == "":
    separation_auto = "N"


if separation_auto == "N":
    if systeme == "D":
        doss_trav = "."
    else:
        print("")
        doss_trav = input("Dans quel dossier se trouvent les sous-dossiers trn, val et tst contenant les fichiers las "
                          "d'entrainement? \n")
else:
    print("")
    doss_trav = input("Dans quel dossier se trouvent les fichiers las d'entrainement? \n")
    las_init = [os.path.join(doss_trav, fich1) for fich1 in os.listdir(doss_trav) if fich1.upper().endswith('.LAS')]
    if not las_init:
        sortie = input("Pas de fichier las dans ce dossier.")
        exit()
    print("")
    nb_trnvaltst = input("Combien de fichiers las doivent servir pour les dossiers trn, val, tst? \n")
    liste_trnvaltst = nb_trnvaltst.replace(" ", "").split(",")
    if len(liste_trnvaltst) != 3:
        sortie = input("On demande 3 chiffres.")
        exit()
    else:
        nb_trn = int(liste_trnvaltst[0])
        nb_val = int(liste_trnvaltst[1])
        nb_tst = int(liste_trnvaltst[2])

    if nb_trn + nb_val + nb_tst > len(las_init):
        sortie = input("Pas de assez de fichiers las dans le dossier.")
        exit()
    else:
        liste_trn = las_init[0:nb_trn]
        liste_val = las_init[nb_trn:nb_trn+nb_val]
        liste_tst = las_init[nb_trn+nb_val:nb_trn+nb_val+nb_tst]

    doss_intrants_trn = os.path.join(doss_trav, "trn")
    if not os.path.exists(doss_intrants_trn):
        os.system("mkdir {0}".format(doss_intrants_trn))

    doss_intrants_val = os.path.join(doss_trav, "val")
    if not os.path.exists(doss_intrants_val):
        os.system("mkdir {0}".format(doss_intrants_val))

    doss_intrants_tst = os.path.join(doss_trav, "tst")
    if not os.path.exists(doss_intrants_tst):
        os.system("mkdir {0}".format(doss_intrants_tst))

    print("")
    print("Déplacement des fichiers las du dossier intrants vers les sous-dossiers trn, val et tst")

    for f1 in liste_trn:
        shutil.move(f1, doss_intrants_trn)

    for f2 in liste_val:
        shutil.move(f2, doss_intrants_val)

    for f3 in liste_tst:
        shutil.move(f3, doss_intrants_tst)


debut = time.time()

print("")
print("Préparation des données des dossiers intrants vers les dossiers datasets")

doss_datasets = os.path.join(doss_trav, "Datasets")
if not os.path.exists(doss_datasets):
    os.system("mkdir {0}".format(doss_datasets))

doss_datasets_trn = os.path.join(doss_datasets, "trn")
if not os.path.exists(doss_datasets_trn):
    os.system("mkdir {0}".format(doss_datasets_trn))

doss_datasets_val = os.path.join(doss_datasets, "val")
if not os.path.exists(doss_datasets_val):
    os.system("mkdir {0}".format(doss_datasets_val))

doss_datasets_tst = os.path.join(doss_datasets, "tst")
if not os.path.exists(doss_datasets_tst):
    os.system("mkdir {0}".format(doss_datasets_tst))

doss_test = os.path.join(doss_trav, "tst")

doss_res = os.path.join(doss_trav, "resultats")
if not os.path.exists(doss_res):
    os.system("mkdir {0}".format(doss_res))

if systeme == "D":
    doss_modele = os.path.join(doss_trav, "models")
else:
    doss_modele = os.path.join(chemin_CP, "models")

if not os.path.exists(doss_modele):
    os.system("mkdir {0}".format(doss_modele))

# doss_datasets_results = os.path.join(doss_datasets, "results")
# if not os.path.exists(doss_datasets_results):
#    os.system("mkdir {0}".format(doss_datasets_results))

# script_prep = r"/home/ubuntu/anaconda2/envs/pytorch_convpoint_p36/ConvPoint/examples/airborne_lidar/
# prepare_airborne_lidar_label.py"
script_prep = os.path.join(chemin_CP, "examples/airborne_lidar/prepare_airborne_lidar_label.py")
os.system("python {0} --folder {1} --dest {2}".format(script_prep, doss_trav, doss_datasets))

print("")
print("Entrainement selon données des dossiers datasets")

# script_train = r"/home/ubuntu/anaconda2/envs/pytorch_convpoint_p36/ConvPoint/examples/airborne_lidar/
# airborne_lidar_seg_wandb.py"
script_train = os.path.join(chemin_CP, "examples/airborne_lidar/airborne_lidar_seg_wandb.py")

os.system("python {0} --savedir {1} --rootdir {2} --testdir {3} --resdir {4} --model_state {5} --dsname {6} --lr {7} "
          "--npoints {8} --blocksize {9} --iter {10} --nepochs {11} --batchsize {12} --num_workers {13} "
          "--finetune {14} --decapit {15} --mode {16}".format(script_train, doss_modele, doss_datasets, doss_test,
                                                              doss_res, modele_pth, nom_entrain, lr0, nbpoints0,
                                                              blocksize0, iter0, nbepochs0, batchsize0, nbworkers0,
                                                              option_finetune, option_decapit, code_classes))

if separation_auto == "O":
    print("")
    print("Redéplacement des fichiers las vers le dossier initial")

    for f4 in os.listdir(doss_intrants_trn):
        shutil.move(os.path.join(doss_intrants_trn, f4), doss_trav)
    for f5 in os.listdir(doss_intrants_val):
        shutil.move(os.path.join(doss_intrants_val, f5), doss_trav)
    for f6 in os.listdir(doss_intrants_tst):
        shutil.move(os.path.join(doss_intrants_tst, f6), doss_trav)

shutil.rmtree(doss_datasets)

fin = time.time()

print("")
print("L'ensemble des traitements a pris {0}".format(temps_ecoul(debut, fin)))
