import requests

url = 'http://127.0.0.1:8000/predict'

headers = {
    'accept': 'application/json',
}
files = {
    'file': ('Test_folder/Grape_Black_rot2.jpg', 
             open('Test_folder/Grape_Black_rot2.jpg', 'rb'), 
             'image/jpeg')
}

response = requests.post(url, headers=headers, files=files)

print(response.status_code)
print(response.json())