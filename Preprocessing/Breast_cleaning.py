import pandas as pd
import pickle as pkl
# read the csv
df = pd.read_csv('../Datasets/dataset_comment_list.csv', sep=';')

couples = [i for i in df['Filename']]

couples = [i.split('&') for i in couples]
duplicated_list = []
for c in couples:
    if len(c) > 1:
        duplicated_list.append(c[0].split('{')[1])
        duplicated_list.append(c[1].split('}')[0])

print('Duplicated images:', len(duplicated_list))


def split_bracketed_values(column):
    extracted = column.str.extract(r'\{(.*?)\}')[0]
    return extracted.str.split('&', expand=True)

new_columns = split_bracketed_values(df['Filename'])

new_columns.columns = [f'image_{i+1}' for i in range(new_columns.shape[1])]
df = pd.concat([df, new_columns], axis=1)
df = df.drop('Filename', axis=1)
df = df[df['image_2'].notnull()] 

image_1 = df['image_1'].tolist()
image_2 = df['image_2'].tolist()
image_3 = df['image_3'].tolist()
image_4 = df['image_4'].tolist()
image_5 = df['image_5'].tolist()
image_6 = df['image_6'].tolist()
image_7 = df['image_7'].tolist()

image_1 = [i for i in image_1 if i]
image_2 = [i for i in image_2 if i]
image_3 = [i for i in image_3 if i]
image_4 = [i for i in image_4 if i]
image_5 = [i for i in image_5 if i]
image_6 = [i for i in image_6 if i]
image_7 = [i for i in image_7 if i]

images_to_delete = image_1 + image_2 + image_3 + image_4 + image_5 + image_6 + image_7

print('Images to delete:', len(images_to_delete))

with open('../Datasets/Breast_US_images_to_delete.pkl', 'wb') as f:
    pkl.dump(images_to_delete, f)
