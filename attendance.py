import pandas as pd
import numpy as np
import os
import datetime


FACE_DATABASE_DIR = './mydatabase'
ATTENDANCE_FILENAME = './attendance/attendance.csv'

CSV_COL_NAME = 'Names'

DATE_TIME_FORMAT = "%d-%m-%Y %H:%M:%S"


class AttendanceMarker:

    def __init__(self):
        # current datetime to put attendance 
        now = datetime.datetime.now()
        self.time = now.strftime(DATE_TIME_FORMAT)

    def _create_new_csv(self):
        names = os.listdir(FACE_DATABASE_DIR)
        names = np.array(names)
        df = pd.DataFrame(data=names,columns=[config.CSV_COL_NAME])
        df.to_csv(ATTENDANCE_FILENAME,index=False)



    def mark_attendance(self,names):

        if(not os.path.exists(ATTENDANCE_FILENAME)):
            self._create_new_csv()

        df = pd.read_csv(ATTENDANCE_FILENAME)
        df[self.time] = 0
        for name in names:
            df.loc[df[CSV_COL_NAME] == name,self.time] = 1

        df.to_csv(ATTENDANCE_FILENAME,index=False)
        print('Saving attendance for  names -> [ {} ] to file :{} at : {}'.format(names, ATTENDANCE_FILENAME,self.time))



