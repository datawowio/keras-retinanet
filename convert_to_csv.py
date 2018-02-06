import pymongo
import os

DATA_DIR = '/home/datawow/data/pictures'  # Pleas change this variable
OUTPUT_FILE = 'dataset_training.csv'

mongo = pymongo.MongoClient('mongodb://192.168.4.12')
face_col = mongo.gender.faces
faces = list(face_col.find({'label': {'$ne': 'undefined'}, 'set': 'training', 'coordinates': {'$ne': None}}))

with open(OUTPUT_FILE, 'w') as csv:
    for f in faces:
        data = [
            os.path.join(DATA_DIR, f['filename']),
            int(f['coordinates']['x1']),
            int(f['coordinates']['y1']),
            int(f['coordinates']['x2']),
            int(f['coordinates']['y2']),
            f['label'],
        ]
        line = ','.join(map(str, data))
        csv.write(line+"\n")
