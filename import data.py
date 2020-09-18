# pandas library
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
import pandas as pd

# csv file
df = pd.read_csv('file_name.csv', index_col=0, header=0)

# text file
df = pd.read_csv('file_name.txt', delimiter='\t')

# excel file
df = pd.read_excel('file_name.xslx')





# numpy library
# https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
from numpy import genfromtxt
my_data = genfromtxt('my_file.csv', delimiter=',')

# csv library
# https://docs.python.org/3/library/csv.html
import csv
with open('winequality-red.csv', 'r') as f:
    wines = list(csv.reader(f, delimiter=';'))


