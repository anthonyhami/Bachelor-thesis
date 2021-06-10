import os
import numpy as np
import pandas as pd


from os import listdir
from os.path import isfile, join


python_code = "USEMODELAC_HP.py"

df = pd.read_csv("HP_combination.csv",sep=";")

print (df)

for i, row in df.iterrows():
    print (i)
    output_dir_name = row["output_dir_name"]

    onlyfiles = [f for f in listdir(output_dir_name) if isfile(join(output_dir_name, f)) and "model_" in f]
    onlyfiles.sort(key=lambda x: int(x[6:-3])) # remove "model_" and ".h5" from the string
    print ("onlyfiles")
    print (onlyfiles)
    print (onlyfiles[-1])
    last_model = onlyfiles[-1]

    # bsub -n 4 -eo error.txt -oo output.txt  python histo.py

    python_command = 'python %s --model_name %s --output_dir_name %s ' % (
        python_code,
        last_model,
        output_dir_name,
        )

    bsub_command = "bsub -n 4 -eo %s_error_USEMODEL.txt -oo %s_output_USEMODEL.txt %s " % (
        row["output_dir_name"],
        row["output_dir_name"],
        python_command,
        )

    print (bsub_command)

    # run bsub_command
    os.system(bsub_command)
