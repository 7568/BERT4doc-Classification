import shutil

import numpy as np
import pandas as pd


def get_adult_data():
    url_data = "/home/liyu/data/tabular-data/adult/adult.data"

    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    label = "income"
    columns = features + [label]
    df = pd.read_csv(url_data, names=columns)

    # Fill NaN with something better?
    df.fillna("don't know" , inplace=True)

    tabular_to_word=[]
    tabular_to_textual=[]
    tabular_with_pump_to_textual=[]
    y=[]
    for index, row in df.iterrows():
        line_to_word = ''
        line_to_content = 'This is an adult person information'
        sex= 'his' if row['sex'].strip().lower()=='Male'.lower() else 'her'
        man_women= 'man' if row['sex'].strip().lower()=='Male'.lower() else 'woman'

        for f in features:
            line_to_word += f',{row[f]}'
            line_to_content += f', and {sex} {f} is {row[f]}'
        tabular_to_word.append([int(row[label].strip()==">50K"),line_to_word[1:]])
        line_to_content += f". The information show that this person's {label} is rich {man_women}."
        tabular_to_textual.append([int(row[label].strip()==">50K"),line_to_content])
        y.append(int(row[label].strip()==">50K"))
        # print(f'row[label] is {row[label].strip()} and row[label]>50K : {row[label].strip()==">50K"}')
    tabular_to_word = np.array(tabular_to_word)
    tabular_to_textual = np.array(tabular_to_textual)
    y = np.array(y)
    cross_validation_data_save('/home/liyu/data/tabular-data/adult/tabular_to_textual/',tabular_to_textual,y)
    cross_validation_data_save('/home/liyu/data/tabular-data/adult/tabular_to_word/',tabular_to_word,y)


from sklearn.model_selection import KFold, StratifiedKFold  # , train_test_split
import os
def remove_file_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def cross_validation_data_save(path,X,y):
    num_splits=5
    objective='binary'
    shuffle=True
    seed=123
    if objective == "regression":
        kf = KFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)
    elif objective == "classification" or objective == "binary":
        kf = StratifiedKFold(n_splits=num_splits, shuffle=shuffle, random_state=seed)
    else:
        raise NotImplementedError("Objective" + objective + "is not yet implemented.")

    remove_file_if_exists(path)
    os.mkdir(path)
    for i, (train_index, test_index) in enumerate(kf.split(X,y)):

        train_data, test_data = X[train_index], X[test_index]
        os.mkdir(path+f'/cross_validation_{i}')
        pd.DataFrame(train_data).to_csv(path+f'/cross_validation_{i}/train.csv',sep="\t", index=False)
        pd.DataFrame(test_data).to_csv(path+f'/cross_validation_{i}/test.csv',sep="\t", index=False)



if __name__ == '__main__':
    get_adult_data()