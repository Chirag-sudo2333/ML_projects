import random
a = str(input("Roll the dice (y/n)")).lower()
while True:
    if a=="y":
        while a =="y":
                b = random.choices((range(1, 7)), k=2) 
                print(b)
                a = str(input("Roll the dice (y/n)")).lower()
    elif a=="n":
        print("Thanks for playing")
        break
    else:
        print("Invalid syntax")
        break

    