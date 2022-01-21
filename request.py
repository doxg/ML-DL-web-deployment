import requests

#response = requests.post("http://127.0.0.1:5000/predict", json=[[10, 15, 0, 0.5]])
files = {'media': open('test.jpg', 'rb')}
response = requests.post("http://127.0.0.1:5000/predict", files=files)
print(response.text)