from bottledaemon import daemon_run
from bottle import BaseRequest, request, run, post, route
from PIL import Image
from StringIO import StringIO
import read_single_image
import time

# aumentar peso maximo de archivo a recibir, en bytes
BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10 # 10 MB max
        
@post('/inference')
def inference():
    t0 = time.time()
    data = request.files.image
        
    raw = data.file.read()
    im = Image.open(StringIO(raw))
    
    response = {
        'lectura': read_single_image.main(im),
        'tiempo_inferencia': time.time() - t0
    }
    
    return response

@route('/test')
def test():
    return("Hello world!")

run(host='0.0.0.0', port=8080, debug=True)

#if __name__ == "__main__":
#  daemon_run()
