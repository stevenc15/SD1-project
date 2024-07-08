import pandas as pd
# import openpyxl

class ArmModel:

    file_path = ""
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.time = None
        self.load_data()
    
    def load_data(self):
        # Load here all the data that is needed from the excel file. For now, I have only the time column loaded
        self.data = pd.read_excel(self.file_path)
        column = self.data['time']
        self.time = column.to_dict()
    
    # def get_data_from_file(self):
    #     # data = pd.read_excel(self.file_path)
        
    #     columns = self.data[['time', 'arm_flex_r', 'arm_add_r', 'elbow_flex_r']]
    #     dic = columns.to_dict(orient='records')
    #     return dic


    def get_data_point(self, idx):
        if idx < len(self.time):
            return self.time[idx]
        else:
            return None

