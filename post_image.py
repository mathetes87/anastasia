import os, requests
folder_path = '/home/mathetes/Desktop/SVHN/Redes/glucometros_clasificacion_end_to_end'
os.chdir(folder_path)

with open('0.jpg', 'rb') as f: 
    r = requests.post(
            'http://13.92.198.82:8080/inference', 
            files={'image': f}
        )
    print r.text