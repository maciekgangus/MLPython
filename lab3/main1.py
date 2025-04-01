import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import time as tm

def_path = os.getcwd()

DataFrame = pd.read_csv(f"{def_path}/Advertising.csv", delimiter=',')
print(DataFrame)
DataFrame.to_csv