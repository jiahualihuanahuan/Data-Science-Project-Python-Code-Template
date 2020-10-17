# pandas library
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
#----------------------------------------
# 1 use high level API like numpy or pandas to read data
#----------------------------------------
# 1.1 pandas library
import pandas as pd


# csv file
df = pd.read_csv('file_name.csv', index_col=0, header=0)

# text file
df = pd.read_csv('file_name.txt', delimiter='\t')

# excel file
df = pd.read_excel('file_name.xslx')

# html table
tables = pd.read_html("http://www.basketball-reference.com/leagues/NBA_2016_games.html")

# json
df = pd.read_json('file_name.json')




#----------------------------------------
# 1.2 numpy library
# https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
from numpy import genfromtxt
my_data = genfromtxt('my_file.csv', delimiter=',')

# csv library
# https://docs.python.org/3/library/csv.html
import csv
with open('winequality-red.csv', 'r') as f:
    wines = list(csv.reader(f, delimiter=';'))
#----------------------------------------

# 1.3 tarfile library
# tar.gz
# https://docs.python.org/3/library/tarfile.html
import tarfile
tar = tarfile.open("sample.tar.gz")
tar.extractall()
tar.close()
#----------------------------------------
#  1.4 json library
# https://docs.python.org/3/library/json.html

import json

file_name = './books.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
#----------------------------------------
# 2 use lower level python build-in functions 
#----------------------------------------
# 2.1 Python build-in functions (read(), readline(), and readlines())

with open('sample.txt', 'r') as reader:
    # Read & print the entire file
    print(reader.read())

with open('sample.txt', 'r') as reader:
    # Read & print the first 5 characters of the file
    print(reader.read(5))

with open('sample.txt', 'r') as reader:
    # Read & print the first 5 characters of the line 1 time
    print(reader.readline(5))

with open('sample.txt', 'r') as reader:
    # Read & print the first 5 characters of the line 5 times
    print(reader.readline(5))
    print(reader.readline(5))
    print(reader.readline(5))
    print(reader.readline(5))
    print(reader.readline(5))

with open('sample.txt', 'r') as reader:
    print(reader.readlines())

# Iterating Over Each Line in the File
with open('sample.txt', 'r') as reader:
    # Read and print the entire file line by line
    line = reader.readline()
    while line != '':  # The EOF char is an empty string
        print(line, end='')
        line = reader.readline()
#----------------------------------------
# 2.2 Python csv module
import csv
with open('sample.csv','r') as myFile:  
    lines=csv.reader(myFile, delimiter=',')  
    for line in lines:  
        print(line)

with open('sample.csv','r') as myFile: 
    lines=csv.DictReader(myFile, delimiter=',')
    for line in lines:
        print(line)

filed1=[]
filed2=[]
filed3=[]
with open('sample.csv','r') as myFile: 
    lines=csv.DictReader(myFile, delimiter=',')
    for line in lines:
        filed1.append(line['Province/State'])
        filed2.append(line['Country/Region'])
        filed3.append(line['Last Update'])
#----------------------------------------
# 3 Examples

# 3.1 use mlxtend library to load mnist dataset
from mlxtend.data import loadlocal_mnist
X, y = loadlocal_mnist(
            images_path='/Users/jiahuali1991/Documents/GitHub/Data-Science-Project-Python-Code-Template/train-images-idx3-ubyte', 
            labels_path='/Users/jiahuali1991/Documents/GitHub/Data-Science-Project-Python-Code-Template/train-labels-idx1-ubyte')

X_test, y_test = loadlocal_mnist(
            images_path='/Users/jiahuali1991/Documents/GitHub/Data-Science-Project-Python-Code-Template/t10k-images-idx3-ubyte', 
            labels_path='/Users/jiahuali1991/Documents/GitHub/Data-Science-Project-Python-Code-Template/t10k-labels-idx1-ubyte')

#----------------------------------------
# 3.2 Python program to convert JSON file to CSV 
  
  
import json 
import csv 
  
  
# Opening JSON file and loading the data 
# into the variable data 
with open('data.json') as json_file: 
    data = json.load(json_file) 
  
employee_data = data['emp_details'] 
  
# now we will open a file for writing 
data_file = open('data_file.csv', 'w') 
  
# create the csv writer object 
csv_writer = csv.writer(data_file) 
  
# Counter variable used for writing  
# headers to the CSV file 
count = 0
  
for emp in employee_data: 
    if count == 0: 
  
        # Writing headers of CSV file 
        header = emp.keys() 
        csv_writer.writerow(header) 
        count += 1
  
    # Writing data of CSV file 
    csv_writer.writerow(emp.values()) 
  
data_file.close() 
#----------------------------------------


