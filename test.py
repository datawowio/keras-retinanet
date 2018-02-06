# show image inline


#automatically reload modules where they have changed


#import keras
import keras

#import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

#import miscellaneous modules
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

#set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)
#use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

model_path =os.path.join('/home/datawow/projects/retina_trial/keras-retinanet/snapshots/resnet50_gender_detection_full_dataset.h5')
model = keras.models.load_model(model_path,custom_objects = custom_objects)
#print model.summary()

#load label to names mapping for visualization purposes
labels_to_names = {0: 'male', 1: 'female'}

path = '/home/datawow/projects/retina_trial/keras-retinanet/examples/test_images'
for filename in os.listdir(path):
    image = read_image_bgr(os.path.join(path, filename))

    draw = image.copy()
    draw = cv2.cvtColor(draw,cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image,scale = resize_image(image)

    start = time.time()
    _,_,detections = model.predict_on_batch(np.expand_dims(image,axis = 0))
    print("processing time: ", time.time() - start)

    predicted_labels = np.argmax(detections[0, :, 4:], axis = 1)
    scores = detections[0,np.arange(detections.shape[1]), 4 + predicted_labels]

    detections[0, :,:4] /= scale

    for idx, (label,score) in enumerate(zip(predicted_labels,scores)):
        if score < 0.5:
            continue
        b = detections[0,idx,:4].astype(int)
        cv2.rectangle(draw,(b[0],b[1]),(b[2],b[3]),(0,0,255),3)
        caption = "{}{:.3f}".format(labels_to_names[label],score)
        cv2.putText(draw,caption,(b[0],b[1]-10),cv2.FONT_HERSHEY_PLAIN,7,(0,0,0),9)
        cv2.putText(draw,caption,(b[0],b[1]-10),cv2.FONT_HERSHEY_PLAIN,7,(255,255,255),6)

    plt.figure(figsize=(15,15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
    plt.savefig('output_'+filename)


