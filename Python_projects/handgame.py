import random

b = ["scissors", "paper", "rock"]
emoji = { 
    "rock" : "🪨",
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

def user_choice():
    while True:
        opp_chc = random.choice(b)
        user_ch = str(input("Enter your choice\n R for rock, P for paper, and S for Scissors: ")).lower()
        if user_ch in lu:
            return user_ch, opp_chc
        else:
            print("Invalid Input!!")

def emoji_value(user_ch, opp_chc):
    user_chc = shrt[user_ch]
    print("You chose " + emoji[user_chc])
    print("Opponent chose " + emoji[opp_chc])
    return user_chc
    
def choose_winner(user_chc, opp_chc):
    if user_chc == opp_chc:
        print("Draw!!")
    elif rules[user_chc] == opp_chc:
        print("You win!!")
    else:
        print("You lose!!")
    
def conti_user():
    x = str(input("Do you want to continue? (y/n): ")).lower()
    if x == "n":
        return False
    elif x != "y":
        print("Invalid Choice.")
        return False
    else:
        return True

def play_game():  
    while True:
        user_ch, opp_chc = user_choice()
        user_chc = emoji_value(user_ch, opp_chc)
        choose_winner(user_chc, opp_chc)
        if not conti_user():
            break

# Main code
play_game()