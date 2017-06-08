#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import load_model
from PIL import Image
import numpy as np
import sys
import argparse

def square_image(im, new_side):
    old_size = im.size
    new_side = int(np.max(old_size))

    new_size = (new_side, new_side)
    new_im = Image.new("RGB", new_size)

    width_added = int((new_size[0]-old_size[0])/2)
    height_added = int((new_size[1]-old_size[1])/2)
    new_im.paste(im, (width_added, height_added))

    return new_im

def main(im):
    # parametros a ocupar
    GLUC_NEW_SIZE = 96
    DIGITS_NEW_SIZE = 48
        
    # transformar imagen
    im = square_image(im, new_side=GLUC_NEW_SIZE)
    im = im.resize((GLUC_NEW_SIZE, GLUC_NEW_SIZE), resample=Image.BICUBIC)
    
    # pasar a formato que recibe el modelo, en numpy
    image = np.expand_dims(np.array(im), axis=0) / 255.

    try:
        reg_model = load_model('regression_model.hdf5')
        clas_model = load_model('classification_model.hdf5')
    except Exception, e:
        print("Error en lectura de modelos: {}".format(e))
        sys.exit()
    
    # obtener bbox
    regression = reg_model.predict(image)[0]
    x1, y1, x2, y2 = regression    
    
    # recortar imagen con bbox y pasar a formato de siguiente modelo
    dig = im.crop([x1, y1, x2, y2])
    dig = square_image(dig, DIGITS_NEW_SIZE)
    dig = dig.resize((DIGITS_NEW_SIZE, DIGITS_NEW_SIZE), resample=Image.BICUBIC)   
        
    digits = np.expand_dims(np.array(dig), axis=0) / 255.
    
    # finalmente encontrar digitos
    classification = clas_model.predict(digits)
    result = ''.join([str(np.argmax(y_)) if np.argmax(y_) != 10 else '' for y_ in classification])
    
    return result

if __name__ == "__main__":
    # recibir argumentos
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, nargs='+',
                    help="ruta de la imagen con su nombre")
                    
    args = vars(ap.parse_args())
    
    image_path = args['image'][0]
    im = Image.open(image_path)

    main(im)
    
    
