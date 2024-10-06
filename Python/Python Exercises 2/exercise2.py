rhyme = 'Twinkle, twinkle, little star. How I wonder what you are!'
char_counts = {}
clean_rhyme = ''.join(char.lower() for char in rhyme if char.isalpha())

for char in clean_rhyme:
  if char in char_counts:
    char_counts[char] += 1
  else:
    char_counts[char] = 1
    
most_frequent_char = max(char_counts, key=char_counts.get)
count = char_counts[most_frequent_char]

print(f"Most frequent character: {most_frequent_char} (appears {count} times)")