import requests
# debugger import, n for next line debug and c for whole code debug
# import pdb; pdb.set_trace()

url = 'https://jsonplaceholder.typicode.com/posts'

response = requests.get(url)

# print(response)
# print(response.status_code)
# print(response.json())

data = response.json()

# print(type(data),len(data))