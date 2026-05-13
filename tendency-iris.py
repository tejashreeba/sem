import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = sns.load_dataset('iris')
df.groupby('species').count()

setosa_d = df[df['species'] == 'setosa']
versi = df[df['species'] == 'versi_color']
virgi = df[df['species'] == 'verginica']

for a in setosa_d:
    try:
        if a in 'species':
            continue
        print()
        print(a)
        print()
        print("mean: ",setosa_d[a].mean())          
        print("median: ",setosa_d[a].median())
        print("25quartile: ",setosa_d[a].quantile(0.25))
    except TryError:
        continue



















