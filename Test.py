import random
random.seed(a=None, version=2)
word = "string"

letters = list(word)

n = random.randint (1,26)

newword = ""

for letter in letters:
    newletter = chr(ord(letter) + n)
    newword += newletter

print(newword) 