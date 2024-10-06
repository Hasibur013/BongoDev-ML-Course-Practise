list1 = [3, 5, 7, 4, 8, 8]
list2 = [4, 9, 8, 7, 1, 1, 13]

common_items = set(list1) & set(list2)

sum_of_common = sum(item for item in common_items)

print(f"Common items: {common_items}")
print(f"Sum of common items: {sum_of_common}")