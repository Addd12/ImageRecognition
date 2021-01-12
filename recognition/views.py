from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from django.http import HttpResponse


height = 224
width = 224
with open('./models/imagenet_classes.json','r') as f:
    label = f.read()

label = json.loads(label)

model_graph = tf.compat.v1.Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/MyModel.h5')

def index(request):
    context={'a': 1}
    return render(request, 'index.html', context)

def recognize(request):
    if request.method == 'POST' and not request.FILES.get('filePath'):
        return HttpResponse('<h1>Please first upload an image</h1>')
    else:
        filePath = request.FILES['filePath']
        fs = FileSystemStorage()
        filePathName = fs.save(filePath.name, filePath)
        filePathName = fs.url(filePathName)

        testimage='.' + filePathName
        img = image.load_img(testimage, target_size=(height, width))
        x = image.img_to_array(img)
        x = x / 255
        x = x.reshape(1, height, width, 3)
        
        with model_graph.as_default():
            with tf_session.as_default():
                prd = model.predict(x)
        import numpy as np
        predictedLabel=label[str(np.argmax(prd[0]))]
        context = {'filePathName': filePathName, 'predictedLabel': predictedLabel[1]}
        return render(request, 'index.html', context)
