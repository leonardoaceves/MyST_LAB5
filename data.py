import pandas as pd

def dataReader(filepath):
 
    data = pd.read_csv(filepath)
    
    return data