#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import load_model, model_from_json
from bottle import BaseRequest, request, run, post, route
from PIL import Image
from StringIO import StringIO
import numpy as np
from time import time, gmtime, strftime
import sys, os, uuid
import tensorflow as tf
from io import BytesIO
import base64
import urllib
import json
import uuid # para generar nombre único, probablemente
import threading
from azure.storage.blob import BlockBlobService, ContentSettings

# aumentar peso maximo de archivo a recibir, en bytes
BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10 # 10 MB max

# para ejecutar funciones en hilos paralelos
# en este caso guardar la imagen en azure blob storage
class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)
 
    def run(self):
        self._target(*self._args)

def square_image(im, new_side):
    old_size = im.size
    new_side = int(np.max(old_size))

    new_size = (new_side, new_side)
    new_im = Image.new("RGB", new_size)

    width_added = int((new_size[0]-old_size[0])/2)
    height_added = int((new_size[1]-old_size[1])/2)
    new_im.paste(im, (width_added, height_added))

    return new_im  

def load_models(model_name):
    json_file = open('split_models/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('split_models/'+model_name+'_weights.h5')
    print("Modelo "+model_name+" cargado en memoria.")

    # cualquier loss para que compile, igual no la ocupa
    loaded_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=[]) 
    graph = tf.get_default_graph()

    return loaded_model, graph

def save_image_to_azure(base64_image, image_name):
	'''
	Guarda en Azure blob storage una imagen.
	
	:param str base64_image:
		La imagen debe venir en base64 como string
	'''
	
	block_blob_service = BlockBlobService(account_name='imagesanastasia', account_key='8WnRqr5RBPwaefcDyS60DTWqWX+nR02Bo4usikWn6UEttKxEUWnS8dy/0+z1hM4oAKToQOy9IjfCRjx7HZiO3A==')

	block_blob_service.create_blob_from_text(
		container_name = 'glucometros',
		blob_name = image_name,
		text = base64_image
	)


# cargar modelos en memoria
global reg_model, clas_model, reg_graph, clas_graph  
reg_model, reg_graph = load_models('regression')
clas_model, clas_graph = load_models('classification')

@post('/inferencev2')
def inferencev2():
    t0 = time()
    
    body = request.body.read()
    parsedBody = urllib.unquote(body).decode('utf8')
    
    jsonObj = json.loads(parsedBody)
    raw = jsonObj['image'] 
    raw = raw.replace('data:image/jpeg;base64,', '')

	# guardar imagen en Azure en hilo paralelo
    image_name = str(uuid.uuid4())
    hilo_paralelo = FuncThread(save_image_to_azure, raw, image_name)
    hilo_paralelo.start()
        
    im = Image.open(BytesIO(base64.b64decode(raw)))
    
    # parametros a ocupar
    GLUC_NEW_SIZE = 96
    DIGITS_NEW_SIZE = 48
        
    # transformar imagen
    im = square_image(im, new_side=GLUC_NEW_SIZE)
    im = im.resize((GLUC_NEW_SIZE, GLUC_NEW_SIZE), resample=Image.BICUBIC)
    
    # pasar a formato que recibe el modelo, en numpy
    image = np.expand_dims(np.array(im), axis=0) / 255.
    
    # obtener bbox
    with reg_graph.as_default():
	    regression = reg_model.predict(image)[0]
	    x1, y1, x2, y2 = regression    
    # recortar imagen con bbox y pasar a formato de siguiente modelo
    dig = im.crop([x1, y1, x2, y2])
    dig = square_image(dig, DIGITS_NEW_SIZE)
    dig = dig.resize((DIGITS_NEW_SIZE, DIGITS_NEW_SIZE), resample=Image.BICUBIC)   
        
    digits = np.expand_dims(np.array(dig), axis=0) / 255.
    
    # finalmente encontrar digitos
    with clas_graph.as_default():
        classification = clas_model.predict(digits)
        result = ''.join([str(np.argmax(y_)) if np.argmax(y_) != 10 else '' for y_ in classification])
	
	hilo_paralelo.join()

	response = {
        'lectura': result,
        'tiempo_inferencia': time() - t0,
        'time_utc': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
        'storage': image_name
    }
    
    return response

@route('/test')
def test():
    return("Hello world!")

run(host='0.0.0.0', port=8080, debug=True)
