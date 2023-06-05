import json
import os
import pandas as pd

path = r"D:\data\archive\animals_224"
root_path = ""
class_to_id = {}
id_to_class = {}
dirs_name = []
for k, value in enumerate(os.listdir(path)):
    print(k,value)
    class_to_id[value] = k
    id_to_class[k] = value
    dirs_name.append(value)

train_file_name, train_file_label = [], []
test_file_name, test_file_label = [], []
for dir_name in dirs_name:
    for root, dirs, files in os.walk(os.path.join(path, dir_name)):
        for id, file in enumerate(files):
            if id < len(files)*0.8:
                train_file_name.append(os.path.join(path, dir_name, file))
                train_file_label.append(dir_name)
            else:
                test_file_name.append(os.path.join(path, dir_name, file))
                test_file_label.append(dir_name)

with open(r'D:\machine_learning\animals_classfication\class_to_number.json', 'w') as f:
    f.write(json.dumps(class_to_id, indent=4))
    f.close()

with open(r'D:\machine_learning\animals_classfication\number_to_class.json', 'w') as f:
    f.write(json.dumps(id_to_class, indent=4))
    f.close()

train_dct = {'file_name': train_file_name, 'label': train_file_label}
data = pd.DataFrame(train_dct)  # index默认为None
data['number_label'] = data['label'].apply(lambda x: class_to_id[x])
data.to_csv(r"D:\machine_learning\animals_classfication\train_data.csv", index=False)

test_dct = {'file_name': test_file_name, 'label': test_file_label}
data = pd.DataFrame(test_dct)  # index默认为None
data['number_label'] = data['label'].apply(lambda x: class_to_id[x])
data.to_csv(r"D:\machine_learning\animals_classfication\test_data.csv", index=False)
