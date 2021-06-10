import os
import random
import itertools
import numpy as np
import pandas as pd

python_code = "AC_GAN_2_0_HP_tune_final.py"

"""
#num_of_layers,list_of_layers,output_dir_name,num_of_batch
#2,"[32, 16]",bla_2,100
#3,"[32, 16, 16]",bla_3,100

df = pd.read_csv("various_models.csv", sep=",")

print (df)
"""

num_of_experiments = 4
num_of_layers_disc = list(np.arange(num_of_experiments))[1:]
num_of_layers_gen = list(np.arange(num_of_experiments))[1:]
num_of_batch = [32, 64, 100, 120] #random.choices([32, 64, 100, 120], k=3num_of_experiments)

df = pd.DataFrame(columns=[
    "num_of_layers_disc",
    "list_of_neurons_disc",
    "num_of_layers_gen",
    "list_of_neurons_gen",
    "output_dir_name",
    "num_of_batch"]
    )
list1_permutations = itertools.permutations(num_of_layers_disc, len(num_of_batch))
print ("list1_permutations = ", list1_permutations)
list1_permutations = list(itertools.product(*[num_of_layers_disc, num_of_layers_gen, num_of_batch]))
print ("list1_permutations = ", list1_permutations)

for i in list1_permutations:
    print (i)
    list_of_neurons_disc_combination = []
    list_of_neurons_gen_combination = []

    # discriminator list of neurons
    for j in range(i[0]): # for loop over the num_of_layers
        list_of_neurons_disc_combination.append(random.choices([128, 64, 32, 16], k=1)[0])
    list_of_neurons_disc_combination.sort(reverse=True)
    print ("list_of_neurons_disc_combination = ", list_of_neurons_disc_combination)

    # generator list of neurons
    for j in range(i[1]): # for loop over the num_of_layers
        list_of_neurons_gen_combination.append(random.choices([128, 64, 32, 16], k=1)[0])
    list_of_neurons_gen_combination.sort(reverse=True)
    print ("list_of_neurons_gen_combination = ", list_of_neurons_gen_combination)

    print (i[0])
    print (str(list_of_neurons_disc_combination))
    print (i[1])
    print (str(list_of_neurons_gen_combination))

    df.loc[len(df)]= [
        i[0],
        str(list_of_neurons_disc_combination),
        i[1],
        str(list_of_neurons_gen_combination),
        "exp_%s" % (len(df)),
        i[2]]
print (list1_permutations)

print (df)
print (df.dtypes)
df.to_csv("HP_combination.csv",sep=";",index=False)

# python AC_GAN_TUNE.py  --num_of_layers 4 --list_of_layers "[64, 32, 32, 16]" --output_dir_name bla_
# bsub -eo errortune.txt -oo outputtune.txt  python AC_GAN_TUNE.py

for i, row in df.iterrows():
    python_command = 'python %s --num_of_layers_disc %d --list_of_neurons_disc "%s" --num_of_layers_gen %d --list_of_neurons_gen "%s" --output_dir_name %s --num_of_batch %s' % (
        python_code,
        row["num_of_layers_disc"],
        row["list_of_neurons_disc"],
        row["num_of_layers_gen"],
        row["list_of_neurons_gen"],
        row["output_dir_name"],
        row["num_of_batch"],
        )

    bsub_command = "bsub -n 4 -eo %s_error.txt -oo %s_output.txt %s" % (
        row["output_dir_name"],
        row["output_dir_name"],
        python_command,
        )

    print (python_command)
    print (bsub_command)

    # run bsub_command
    os.system(bsub_command)


