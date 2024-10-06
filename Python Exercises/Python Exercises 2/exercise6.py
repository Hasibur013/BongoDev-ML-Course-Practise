def calculate_average(marks):
  if any(mark < 40 for mark in marks):
    return "FAILED"
  else:
    return sum(marks) / len(marks)

student_1 = [40, 35, 70, 90, 56]
student_2 = [57, 35, 80, 98, 46]

average_1 = calculate_average(student_1)
average_2 = calculate_average(student_2)

print(f"Student 1 average: {average_1}")
print(f"Student 2 average: {average_2}")