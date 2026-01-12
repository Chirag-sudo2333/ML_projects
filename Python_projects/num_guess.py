import random

ans_value = random.choice(range(1,101))
def guess():
    while True:
        user_value = input("Enter your guess(1 to 100): ")
        if user_value.isdigit():
            if int(user_value) == ans_value:
                print("Congratulations!! You win.")
                break
            elif 101 > int(user_value) > ans_value:
                print("Your guess is high. ")
            elif 0 < int(user_value) < ans_value:
                print("Your guess is low. ")
            else:
                print("Invalid Input!!")
        else:
            print("That's not number!!")
            
guess()
        
    
