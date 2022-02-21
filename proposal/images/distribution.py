import json
import matplotlib.pyplot as plt
import pandas as pd

with open('file_list.json', 'r') as f:
  data = json.load(f)

xAxis = [key for key, value in data.items()]
yAxis = [len(value) for key, value in data.items()]

fig = plt.figure()
plt.bar(xAxis, yAxis)
plt.xlabel('Number of items per image')
plt.ylabel('Number of images')

plt.show()


# fig = plt.figure()
# ax = fig.add_axes([0,0,1,1])
# langs = ['1', '2', '3', '4', 'PHP']
# students = [23,17,35,29,12]
# ax.bar(langs,students)
# plt.show()