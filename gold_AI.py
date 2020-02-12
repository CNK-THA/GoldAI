import matplotlib.pyplot as plt
import csv
import numpy as np
import datetime

from sklearn import linear_model


values = {}
line_count = 0
day = 0

x = []
y = []

date_1 = datetime.datetime(1991, 6, 21)

with open('gold.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if(row[3] != '' and row[3] != '-1'):
            line_count += 1
            x.append(day)
            y.append(row[3])
            values[day] = row[3]
            day += 1

    print(f'Processed {line_count} lines.')


with open('gold-Current.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if(row[3] != '' and row[3] != '-1'):
            line_count += 1
            x.append(day)
            y.append(row[3])
            values[day] = row[3]
            day += 1
    print(f'Processed {line_count} lines.')

# plt.scatter(x, y, color='red')
# plt.show()


# print(x)
# print(np.array(x).reshape(-1, 1))
regr = linear_model.LinearRegression()
regr.fit(np.array(x).reshape(-1, 1), y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
# -1 because it will add 1 extra day (beginning day)
end_date = date_1 + datetime.timedelta(days=11000)
print('I am predicting on', end_date.strftime("%d %m %Y"), "price is:", regr.predict(
    np.array([11001]).reshape(-1, 1)))  # no need to -1 here. Put in the day you want


#print(end_date.strftime("%d %m %Y"))
