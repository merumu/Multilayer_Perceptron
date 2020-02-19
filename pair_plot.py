import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from FileLoader import FileLoader

def pairPlot(data):
    try:
        df1 = data.drop(columns=['Index','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'])
        df2 = data.drop(columns=['Index','1','2','3','4','5','6','7','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30'])
        df3 = data.drop(columns=['Index','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','24','25','26','27','28','29','30'])
        df4 = data.drop(columns=['Index','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
    except:
        print("Error: wrong column name in data")
        exit()
    sns.set(style="ticks")
    g = sns.PairGrid(df1, hue="diagnosis", height=1)
    g.map_diag(plt.hist, alpha=0.8)
    g.map_offdiag(plt.scatter, s=1)
    g.add_legend()
    h = sns.PairGrid(df2, hue="diagnosis", height=1)
    h.map_diag(plt.hist, alpha=0.8)
    h.map_offdiag(plt.scatter, s=1)
    h.add_legend()
    i = sns.PairGrid(df3, hue="diagnosis", height=1)
    i.map_diag(plt.hist, alpha=0.8)
    i.map_offdiag(plt.scatter, s=1)
    i.add_legend()
    j = sns.PairGrid(df4, hue="diagnosis", height=1)
    j.map_diag(plt.hist, alpha=0.8)
    j.map_offdiag(plt.scatter, s=1)
    j.add_legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 2:
        loader = FileLoader()
        data = loader.load(str(sys.argv[1]))
        pairPlot(data)
    else:
        print("Usage : python pair_plot.py path.csv")