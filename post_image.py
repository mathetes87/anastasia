import requests

with open('0.jpg', 'rb') as f: 
    r = requests.post(
            'http://localhost:8080/inference', 
            files={'image': f}
        )
    print r.text
