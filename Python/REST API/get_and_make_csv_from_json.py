import requests
import csv
import os
import json

# Define the path to the local post.json file
json_path = os.path.join('REST API', 'post.json')

def get_data():
    # Read data from the local post.json file
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"The file {json_path} was not found.")
        data = []
    
    posts = []
    for post in data:
        post_dict = {
            'Id': post['id'],
            'Title': post['title']
        }
        posts.append(post_dict)
    
    return posts

def write_csv_file():
    post_data = get_data()

    # define the path to save the CSV file in the exercise folder
    file_path = os.path.join('REST API', 'posts.csv')
    # open a csv file to write data
    with open(file_path, 'w', newline='') as title_data:
        # create csv writer object
        csv_writer = csv.writer(title_data)
        
        count = 0
        for title in post_data:
            if count == 0:
                header = title.keys()
                csv_writer.writerow(header)
                count += 1
                
            csv_writer.writerow(title.values())

write_csv_file()