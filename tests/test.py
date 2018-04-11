import csv

with open('data/train_in.csv', 'r', newline='\n') as f:
	reader = csv.reader(f)
	my_list = list(f)
	my_list = [i.split(',') for i in my_list]
	t = []
	for l in my_list:
	    t.append([float(i) for i in l])
	my_list = t

	print(type(my_list[0][0]))
