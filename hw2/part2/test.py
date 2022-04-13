
import json
import csv
in_filename = "clean.csv"

with open(in_filename) as in_file:
    for line in in_file:
        small_list = [int(x) for x in line.split(',')]
        break

with open('./p2_data/annotations/train_annos.json', 'r') as f :
        data = json.load(f)    
    
images, categories = data['images'], data['categories']


cnt = 0
for i in small_list:
    path = images[i]
    if '99' in path:
        cnt += 1

print(cnt)

dirty_index_list = []
with open(in_filename) as in_file:
    for line in in_file:
        small_list = [int(x) for x in line.split(',')]
    for s in small_list:
        dirty_index_list.append(images[s])

print(dirty_index_list[:20])