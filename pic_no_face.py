import pymongo
import os

DATA_DIR = '/home/datawow/data/pictures'  # Pleas change this variable
OUTPUT_FILE = 'train_with_negative.csv'

mongo = pymongo.MongoClient('mongodb://192.168.4.12')
# no_pics_col = mongo.instagram.pics
faces_col = mongo.instagram.faces
# no_faces = list(no_pics_col.find({'faces': {'$size': 0}}))
faces = list(faces_col.find({'label':{'$ne':'undefined'}, 'set':'training'}))

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
    # for f in no_faces[:10000]:
    #     data = [
    #         os.path.join(DATA_DIR, f['filename']),
    #         '',
    #         '',
    #         '',
    #         '',
    #         '',
    #     ]
    #     line = ','.join(map(str,data))
    #     csv.write(line+"\n")
