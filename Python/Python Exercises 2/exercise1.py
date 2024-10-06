data_list = [13, 24, 'Karim', {'name': 'guru'}, 45, 17]
new_list = []

for item in data_list:
    if type(item) == int:
        new_list.append(item)

print(new_list)