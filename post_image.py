import requests

with open('0.jpg', 'rb') as f: 
    r = requests.post(
            #'http://localhost:8080/inference', 
            'http://13.92.198.82:8080/inference', 
            files={'image': f}
        )
    print r.text
