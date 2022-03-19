import os
import pandas as pd
#DATA_ROOT = "../daon_data/ishikawa_test"
DATA_ROOT = "../daon_data/ishikawa_test_small_wave"

def list_of_dirs(data_root):
    print("inside list of dirs")
    print(data_root)
    #print(os.listdir(data_root))
    file_name = []
    list2 = []
    list3 = []
    list4 = []
    for file in sorted(os.listdir(data_root)):
        # fname = file.split("_")
        # print(fname)
        print(file)
        file_name.append(file)
    data = {'fname': file_name}
    df = pd.DataFrame(data)
    print(df)
    df.to_csv('test_ishikawa_wave.csv', encoding='utf-8')
        

if __name__ == "__main__":
    print("Start!")
    list_of_dirs(DATA_ROOT)