import os

for indices in range(2, 5):
    for num in range(1, 10, 2):
        sigma = (round(num*(10**(-indices)), indices))
        os.system('/usr/local/bin/python3 /Users/Toan/5ETI/2122_5imi_projet_SR/src/main.py --savesteps --iterations 1000 --sigma ' + str(sigma))