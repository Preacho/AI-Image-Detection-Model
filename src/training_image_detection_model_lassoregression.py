import pandas as pd 

data = pd.read_csv("testing_data.csv")
print(data['images'][0][1] )