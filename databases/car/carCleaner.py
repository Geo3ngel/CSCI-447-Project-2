import re
fileboi = open('./car/car.data','r')
newCar = open('./car/newCarBoi.data','w+')
for line in fileboi:
    copy = line
    copy = re.sub(r'5more',"5",copy)
    copy = re.sub(r'more',"5",copy)
    newCar.write(copy)