import csv

# Read csv
with open('male.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in csvreader:
        em = row[0].split(',')
        print(', '.join(row))
        break