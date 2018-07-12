from xlrd import open_workbook, xldate_as_tuple

def to_table(filename):
    wb = open_workbook(filename)
    sheets = wb.sheets()
    assert(len(sheets) == 1)
    
    s = sheets[0]    
        
    firstrow = []
    for col in range(s.ncols):
        value = s.cell(0, col).value
        firstrow.append(value)
    
    table = []

    for row in range(1,s.nrows):
        values = []
        for col in range(s.ncols):
            value = s.cell(row, col).value
            values.append(value)
        table.append(values)
    
    return table, firstrow    
    
def times(filename, score_type="all"):
    wb = open_workbook(filename)
    sheets = wb.sheets()
    assert(len(sheets) == 1)
    
    s = sheets[0]    
        
    firstrow = []
    for col in range(s.ncols):
        value = s.cell(0, col).value
        firstrow.append(value)
    
    table = []

    for row in range(1,s.nrows):
        values = []
        for col in range(s.ncols):
            value = s.cell(row, col).value
            if score_type == "all":
                values.append(xldate_as_tuple(value, wb.datemode))
            elif score_type == "rightleft":
                if col == 0:
                    values.append(xldate_as_tuple(value, wb.datemode))
                elif col == 1:
                    values.append(value)

        if len(values) == 1:
            values = values[0]
        table.append(values)
    
    return table, firstrow    
    
if __name__ == "__main__":
    #files = ['left_turning_car_bike_conflict.xlsx', 'pedestrian_conflicts.xlsx', 'right_turning_car_bike_conflict.xlsx']
    #excel_basepath = '/data/sweden/excel/'
    
    files = ['Right_Left_Swe_1.xlsx']
    excel_basepath = '/data/sweden2/excel/'
    tables = [times(excel_basepath+x, score_type="rightleft")[0] for x in files]
    
    for table in tables:
        for row in table:
            print(row[0],row[1])
