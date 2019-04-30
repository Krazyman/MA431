from datetime import datetime

# a Rain contains a date, max temperature, min temperature, average temperature,
# and precipitation
# - Rain(date, max_temp, min_temp, avg_temp, precipitation)

class Rain:

    def __init__(self, date, max_temp, min_temp, avg_temp, precipitation):
        self.date = datetime.strptime(date, "%m/%d/%Y") # date
        self.max_temp = data_checker(max_temp) # number
        self.min_temp = data_checker(min_temp) # number
        self.avg_temp = data_checker(avg_temp) # number
        # self.precipitation = str(precipitation)
        self.precipitation = data_checker(precipitation) # number

    def __eq__(self, other):
        return type(other) == Rain and \
               self.date == other.date and \
               self.max_temp == other.max_temp and \
               self.min_temp == other.min_temp and \
               self.avg_temp == other.avg_temp and \
               self.precipitation == other.precipitation 

    def __repr__(self):
        return "{!r} - {!r}\n".format(self.date, self.precipitation)



# .csv -> List
# Reads the rain .cvs file then turn all data into Rain objects.
def read_rain_from_file(filename):
    inFile = open(filename, "r")
    rains = []
    z = inFile.readlines()
    inFile.close()
    for i in range(1, len(z)):
        info = []
        info.append(z[i].split(","))
        n = [info[0][0], info[0][1], info[0][2], info[0][3], info[0][4].strip('"\n"')]
        new_info = Rain(*n)
        rains.append(new_info)
    return rains

# str -> str or float
# checks if data is float or str and returns the correct data type
def data_checker(x):
    try: 
        return float(x)
    except ValueError:
        return x

def data_filter(rains, key, value):
    new_rains = []
    for rain in rains:
        if key == "month":
            if rain.date.month == value:
                new_rains.append(rain)
        elif key == "year":
            if rain.date.year == value:
                new_rains.append(rain)
        elif key == "day":
            if rain.date.day == value:
                new_rains.append(rain)
        elif key == "date":
            if rain.date == datetime.strptime(value, "%m/%d/%Y"):
                new_rains.append(rain)
        elif key == "temp":
            if rain.max_temp == float(value) or rain.min_temp == float(value)\
            or rain.avg_temp == float(value):
                new_rains.append(rain)
    return new_rains

def precipitation_filter(rains, key, value):
    new_rains = []
    value = float(value)
    for rain in rains:
        if rain.precipitation != "T" and rain.precipitation != "M":
            if key == ">":
                if rain.precipitation > value:
                    new_rains.append(rain)
            elif key == ">=":
                if rain.precipitation >= value:
                    new_rains.append(rain)
            elif key == "<":
                if rain.precipitation < value:
                    new_rains.append(rain)
            elif key == "<=":
                if rain.precipitation <= value:
                    new_rains.append(rain)
            elif key == "=":
                if rain.precipitation == value:
                    new_rains.append(rain)
    return new_rains

def counter(rains):
    return len(rains)

rains = read_rain_from_file("testfile.csv")
# a = precipitation_filter(rains, ">=", 0.25)
# b = data_filter(a, "month", 5)
# print(b)
# print(counter(b))
print(rains)