import csv
import numpy as np
# Read csv
with open('male.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    a = []
    wait = 10
    for row in csvreader:
        wait -= 1
        em = [float(i) for i in row[0].split(',')]
        a.append(em)
        print(', '.join(row))
        if wait == 0:
            break