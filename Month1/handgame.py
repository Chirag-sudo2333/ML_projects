import random
b = ["scissors", "paper", "rock"]
emoji = { "rock" : "🪨",
         "paper" : "📄",
         "scissors" : "✃"
         }
shrt = {
    "r" : "rock",
    "p" : "paper",
    "s" : "scissors"
}
rules = {
    "rock" : "scissors",
    "paper" : "rock",
    "scissors" : "paper"
}
lu = ('r', 'p','s')
def game():
    while True:
        opp_chc = random.choice(b)
        user_ch = str(input("Enter your choice\n R for rock, P for paper, and S for Scissors ")).lower()
        if user_ch in lu:
            user_chc = shrt[user_ch]
            print("You chose " + emoji[user_chc])
            print("Opponent chose " +emoji[opp_chc])
            if user_chc == opp_chc:
                print("Draw!!")
            elif rules[user_chc] == opp_chc:
                print("You win!!")
            else:
                print("You lose!!")
        else:
            print("Inavlid Input!!")
        x = str(input("Do you want to continue? (y/n) ")) 

        if x == "n":
                    break
        elif x!= "y":
                    print("Invalid Choice.")
            
game()