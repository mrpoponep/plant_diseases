import requests

url = 'http://127.0.0.1:8000/predict'

headers = {
    'accept': 'application/json',
}

files = {
    'file': ('Test_folder/after_out.jpg', 
             open('Test_folder/after_out.jpg', 'rb'), 
             'image/jpeg')
}

response = requests.post(url, headers=headers, files=files)

print(response.status_code)
print(response.json())