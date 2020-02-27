import pandas as pd
import random
import ast
import numpy as np
import calendar, random
from datetime import datetime

np.random.seed(0)
# All crypto series are observed until 31th of January 2019 (2019-12-31)
# It takes the starting and ending date and "number_samples" entries choosed randomly

def generate_testset(start_date, end_date,output_path):
    file_to_write= open(output_path + start_date + "_" + end_date + ".txt", 'w')
    start=datetime.strptime(start_date, '%Y-%m-%d')
    end=datetime.strptime(end_date, '%Y-%m-%d')

    #get the number of months between two dates
    num_months=(end.year - start.year) * 12 + end.month - start.month
    num_months=num_months+1

    test_set=[]
    for i in range(0,num_months):
        random_day=randomdate(start.year, start.month)

        #if the last random day generated is upper than the last available day, by default it will be set up to the last available day
        #Example:
        # end date: 18-01-2019, thus last available day is: 18.
        #random date generated: 19-01-2019
        #thus, the random date day will be 18 instead of 19.
        if i+1==num_months and random_day.day>end.day:
            random_day=random_day.replace(day=end.day)

        #adding the random day to the list
        file_to_write.write(str(random_day)+"\n")
        #test_set.append(str(random_day))

        #update the new start date and end date (it generates a date per months!)
        new_year=start.year
        new_month=start.month+1 #find a new date in the next month
        if(new_month==13):
            new_month=1
            new_year=start.year+1 #find a new date in the next year
        start=start.replace(year=new_year,month=new_month)

    #file_to_write.write(str(test_set))
    file_to_write.close()
    return

def randomdate(year, month):
    #The itermonthdates() method returns an iterator for the month (1-12) in the year.
    #This iterator will return all days (as datetime.date objects) for the month and all days before the start of the month
    #or after the end of the month that are required to get a complete week.
    dates = calendar.Calendar().itermonthdates(year, month)
    #get only the dates of our month of interest
    dates_of_the_month=[date for date in dates if date.month == month]
    return random.choice(dates_of_the_month)


def get_testset(path_file):
    test_set=[]
    with open(path_file) as td:
        test_dates= td.readlines()

    for test_date in test_dates:
        test_set.append(test_date.replace("\n",""))

    return test_set