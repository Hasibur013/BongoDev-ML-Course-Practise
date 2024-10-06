import requests
import csv
import os

url = 'https://jsonplaceholder.typicode.com/users'

def get_data():
    response = requests.get(url)
    data = response.json() if response.status_code == 200 else []
    
    if len(data):
        users = [] # make csv like.... name|email
        for user in data:
            # print(user['name'])
            user_dict = {}
            user_dict['Name'] = user['name']
            user_dict['Email'] = user['email']
            users.append(user_dict)
        
        return users

def write_csv_file():
    user_data = get_data()

    # define the path to save the CSV file in the exercise folder
    file_path = os.path.join('REST API', 'users.csv')
    # open a csv file to write data
    with open(file_path, 'w', newline='') as employee_data:
        # create csv writer object
        csv_writer = csv.writer(employee_data)
        
        count = 0
        for employee in user_data:
            if count == 0:
                header = employee.keys()
                csv_writer.writerow(header)
                count += 1
                
            csv_writer.writerow(employee.values())

write_csv_file()