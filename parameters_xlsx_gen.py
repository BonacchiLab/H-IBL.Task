""" This script generates the parameters.xlsx file for the full task. It will generate
    a full task (which includes the training trials)with the following parameters randomly generated:
    - position
    - grating

"""


import xlsxwriter

# Generate empty file
workbook = xlsxwriter.Workbook("components_parameters.xlsx")

#Debugging: Check if the workbook was created successfully
if workbook is None:
    raise Exception("Failed to create workbook")
else:
    print("Workbook created successfully")


