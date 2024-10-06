
def div(number):
    if number%3==0 and number%5==0:
        return "FizzBuzz"
    elif number%3==0:
        return "Fizz"
    elif number%5==0:
        return "Buzz"
    else:
        return "Not a FizzBuzz number"
    
number = int(input("Enter a number: "))
print(div(number))