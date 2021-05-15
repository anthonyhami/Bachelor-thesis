import os
import random
import itertools
import numpy as np
import pandas as pd

python_code = "AC_GAN_TUNE.py"

"""
#num_of_layers,list_of_layers,output_dir_name,num_of_batch
#2,"[32, 16]",bla_2,100
#3,"[32, 16, 16]",bla_3,100

df = pd.read_csv("various_models.csv", sep=",")

print (df)
"""

num_of_experiments = 10
num_of_layers = list(np.arange(num_of_experiments))[1:]
num_of_batch = [32, 64, 100, 120] #random.choices([32, 64, 100, 120], k=3num_of_experiments)

df = pd.DataFrame(columns=["num_of_layers", "list_of_layers", "output_dir_name", "num_of_batch"])
list1_permutations = itertools.permutations(num_of_layers, len(num_of_batch))
list1_permutations = list(itertools.product(*[num_of_layers, num_of_batch]))
for i in list1_permutations:
    print (i)
    new_list = []
    for j in range(i[0]): # for loop over the num_of_layers
        new_list.append(random.choices([128, 64, 32, 16], k=1)[0])
    new_list.sort(reverse=True)
    print ("new_list = ", new_list)

    print (i[0])
    print (str(new_list))
    print (i[1])

    df.loc[len(df)]= [i[0], str(new_list), "exp_%s" % (len(df)),  i[1]]
print (list1_permutations)

print (df)

# python AC_GAN_TUNE.py  --num_of_layers 4 --list_of_layers "[64, 32, 32, 16]" --output_dir_name bla_
# bsub -eo errortune.txt -oo outputtune.txt  python AC_GAN_TUNE.py

for i, row in df.iterrows():
    python_command = 'python %s --num_of_layers %d --list_of_layers "%s" --output_dir_name %s --num_of_batch %s' % (
        python_code,
        row["num_of_layers"],
        row["list_of_layers"],
        row["output_dir_name"],
        row["num_of_batch"],
        )

    bsub_command = "bsub -eo %s_error.txt -oo %s_output.txt %s" % (
        row["output_dir_name"],
        row["output_dir_name"],
        python_command,
        )

    print (python_command)
    print (bsub_command)

    # run bsub_command
    os.system(bsub_command)

    
