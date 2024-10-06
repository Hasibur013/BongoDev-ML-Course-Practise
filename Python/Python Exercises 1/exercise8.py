def guessingGame(): 
    number = 6
    user_input = int(input("Enter a number between 1 and 9: "))
    
    if user_input<number:
        print("your guess is almost there")
    elif user_input>number:
        print("your guess is higher")
    else:
        print("Your Guess Is Correct!")
    

guessingGame()