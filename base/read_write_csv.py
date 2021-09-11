import csv
f = csv.reader(open('G:/AIData/TianChi/APTOS/TrainingAnnotation.csv','r'))
row = 0
for i in f:
    print(i)
    row += 1
    if row > 10:
        break


import codecs
data = [
    ("测试1",'软件测试工程师'),
    ("测试2",'软件测试工程师'),
    ("测试3",'软件测试工程师'),
    ("测试4",'软件测试工程师'),
    ("测试5",'软件测试工程师'),
]
#f = open('222.csv','w') #有乱码
f = codecs.open('222.csv','w','gbk')
writer = csv.writer(f)
for i in data:
    writer.writerow(i)
f.close()
