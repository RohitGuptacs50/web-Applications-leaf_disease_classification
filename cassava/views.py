from django.conf.urls import url
from django.shortcuts import render

# Create your views here.


from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
from tensorflow.python.eager.context import context

from efficientnet.tfkeras import EfficientNetB4



img_height, img_width = 512, 512
with open('./models/label_num_to_disease_map.json', 'r') as f:
    labelInfo = f.read()


labelInfo = json.loads(labelInfo)


model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()

    with tf_session.as_default():
        model = load_model('./models/basic_EfNetB0_imagenet_512.h5')


def index(request):
    context = {'a':1}
    return render(request, 'index.html', context)




def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1, img_height, img_width, 3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)

    import numpy as np
    predictedLabel = labelInfo[str(np.argmax(predi[0]))]
    #print(predictedLabel[0])

    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
    return render(request, 'index.html', context)


