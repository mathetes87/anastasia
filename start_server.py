from bottle import BaseRequest, request, run, post
from PIL import Image
from StringIO import StringIO
import read_single_image

# aumentar peso maximo de archivo a recibir, en bytes
BaseRequest.MEMFILE_MAX = 1024 * 1024 * 10 # 10 MB max
        
@post('/inference')
def inference():
    data = request.files.image
        
    raw = data.file.read()
    im = Image.open(StringIO(raw))
    
    response = {
        'lectura': read_single_image.main(im)
    }
    
    return response

run(host='localhost', port=8080, debug=True)