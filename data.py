#%%#
### Step-1 ######################################################################
### Read the json file

import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

data_1 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/7CT_s5_v7-7766.json')
data_2 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/7CT_s6_v6-9122.json')
data_3 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/11CT_s5_v13-6106.json')
data_4 = read_json_file('/Users/ttonny0326/GitHub_Project/neural-network-and-deep-learning-for-combinatorial-optimisation/11CT_s12_v34-6729.json')

## Print the data form the json file 
print("-------------------------------------------------")
print("Data from the json file")
print("-------------------------------------------------")
print(data_1)
print("------------------------------------------------- \n")
print(data_2)
print("------------------------------------------------- \n")
print(data_3)
print("------------------------------------------------- \n")
print(data_4)


# %%
