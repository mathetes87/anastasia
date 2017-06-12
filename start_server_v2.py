from bottledaemon import daemon_run
from keras.models import load_model, model_from_json
from bottle import BaseRequest, request, run, post, route
from PIL import Image
from StringIO import StringIO
import numpy as np
import time, sys, os
import tensorflow as tf

# aumentar peso maximo de archivo a recibir, en bytes
BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10 # 10 MB max

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

# cargar modelos en memoria
global reg_model, clas_model, reg_graph, clas_graph  
reg_model, reg_graph = load_models('regression')
clas_model, clas_graph = load_models('classification')
    
@post('/inference')
def inference():
    print("Inference")
    t0 = time.time()
    data = request.files.image
        
    raw = data.file.read()
    im = Image.open(StringIO(raw))
    
    # parametros a ocupar
    GLUC_NEW_SIZE = 96
    DIGITS_NEW_SIZE = 48
        
    # transformar imagen
    im = square_image(im, new_side=GLUC_NEW_SIZE)
    im = im.resize((GLUC_NEW_SIZE, GLUC_NEW_SIZE), resample=Image.BICUBIC)
    
    # pasar a formato que recibe el modelo, en numpy
    image = np.expand_dims(np.array(im), axis=0) / 255.
    
    # obtener bbox
    print("Regression")
    with reg_graph.as_default():
        print "1"
	regression = reg_model.predict(image)[0]
        print "2"
	x1, y1, x2, y2 = regression    
    # recortar imagen con bbox y pasar a formato de siguiente modelo
    print "3"
    dig = im.crop([x1, y1, x2, y2])
    print "4"
    dig = square_image(dig, DIGITS_NEW_SIZE)
    print "5"
    dig = dig.resize((DIGITS_NEW_SIZE, DIGITS_NEW_SIZE), resample=Image.BICUBIC)   
        
    digits = np.expand_dims(np.array(dig), axis=0) / 255.
    
    # finalmente encontrar digitos
    print("Classification")
    with clas_graph.as_default():
        classification = clas_model.predict(digits)
        result = ''.join([str(np.argmax(y_)) if np.argmax(y_) != 10 else '' for y_ in classification])
    response = {
        'lectura': result,
        'tiempo_inferencia': time.time() - t0
    }
    
    print("Return")
    return response

@route('/test')
def test():
    return("Hello world!")

run(host='0.0.0.0', port=8080, debug=True)

#if __name__ == "__main__":
#  daemon_run(host='0.0.0.0', port=8080)