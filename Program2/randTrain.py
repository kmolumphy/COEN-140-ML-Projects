import sys
import random
file_path = 'ans.txt'
sys.stdout = open(file_path, "w")
options = [-1,1]
for x in range(392):
    print(random.choice(options))