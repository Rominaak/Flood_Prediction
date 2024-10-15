import pandas as pd
data = pd.read_csv('MalaysiaFloodDataset.csv')# reading the csv file
y = data.iloc[1:825,4:-1].values #assigning the 824 rows and column 4-15 to x

#because the range of the data is large,
#the whole data will have a value of 0-1
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(y)
x = mms.transform(y)

from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, max_iter=300).fit(y) #n_clusters = 2(the alarm is either on or off, so there are only 2 scenarios), maximum iteration(max_iter) is 300 by default
a = model.labels_ #putting model labels (0 or 1)as a list into a
b = data.iloc[1:825,0].values #putting the first column as a list into b

# creating a list of column names
column_values = ['yes/no']

# creating the dataframe
df = pd.DataFrame(data = a,index = b, columns = column_values)

# displaying the dataframe
print(df)

# writing to excel
writer = pd.ExcelWriter('Flood_Result(written from python).xlsx',engine ='xlsxwriter')
df.to_excel(writer,sheet_name = 'test result')
writer.save()
