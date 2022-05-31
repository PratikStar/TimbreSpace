import csv
import numpy as np
# Read csv
with open('male.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    a = np.array((128,))
    for row in csvreader:
        em = [float(i) for i in row[0].split(',')]
        np.insert(a, em)
        print(', '.join(row))
        break