import os

path=r"./REFERENCE.csv"
save_path=r"./data/"


data1_filename = open(r"./data/label_2.csv", "a")
data1_filename.write("Recording,First_label,Second_label,Third_label\n")

for i, line in enumerate(open(path)):
    if i < 2:
        continue
    else:
        strs = line.strip().split(",")
        strs = list(filter(bool, strs))
        data_name = strs[0]
        data_label=int(strs[1])

        if data_label==2:
            print(data_label)
            data1_filename.write("{},{},,\n".format(data_name,data_label))
data1_filename.close()