dict1 = {'age': 13, 'id': 12, 'address': 'Banani', 'course': 'Python'}
dict2 = {'address': 'Rupnagar', 'id': 25, 'course': 'MERN'}

common_keys = set(dict1.keys()) & set(dict2.keys())

print("Common key-value pairs:")
for key in common_keys:
  print(f"\t{key}: {dict1[key]} ({dict2[key]})")