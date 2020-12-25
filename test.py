import json

with open('testdata.json', 'r') as f:
    for line in f.readlines():
        data = json.loads(line)
        print('ere')
print('done')