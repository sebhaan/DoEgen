"""
Generates Excel file with format for experimental design setup

Author: Sebastian Haan
Affiliation: Sydney Informatics Hub (SIH), THe University of Sydney
Version: 0.1
License: APGL-3.0
"""

import xlsxwriter

workbook = xlsxwriter.Workbook('Experiment_setup_template.xlsx')
worksheet = workbook.add_worksheet()

workbook.set_properties({
    'title':    'Experimental Design Setup',
    'subject':  'Template',
    'author':   'Sebastian Haan',
    'company':  'SIH, The University of Sydney',
    'comments': 'Created with Python and XlsxWriter'
})

# Add a format for the header cells.
header_format = workbook.add_format({
    'border': 2,
    'bg_color': '#C6EFCE',
    'bold': True,
    'text_wrap': True,
    'valign': 'bottom',
    'indent': 1,
    'locked': True
})

unlocked = workbook.add_format({'locked': False})
# Enable worksheet protection
#worksheet.protect(options={'autofilter': True})
#worksheet.autofilter('A1:B8')

worksheet.protect()

header_format.set_font_size(14)

worksheet.set_default_row(20)
worksheet.set_row(0, 30)

worksheet.set_column('A:E', None, unlocked)

# Set up layout of the worksheet.
worksheet.set_column('A:A', 50, unlocked)
worksheet.set_column('B:B', 20, unlocked)
worksheet.set_column('C:C', 20, unlocked)
worksheet.set_column('D:D', 20, unlocked)
worksheet.set_column('E:E', 20, unlocked)


# Write the header cells and some data that will be used in the examples.
heading1 = 'Parameter Name'
heading2 = 'Parameter Type'
heading3 = 'Level Number'
heading4 = 'Minimum'
heading5 = 'Maximum'

worksheet.write('A1', heading1, header_format)
worksheet.write('B1', heading2, header_format)
worksheet.write('C1', heading3, header_format)
worksheet.write('D1', heading4, header_format)
worksheet.write('E1', heading5, header_format)

#worksheet.write_row('B2:B10', ['Continous', 'Discrete', 'Categorical'])
worksheet.data_validation('B2:B20', {'validate': 'list',
                                  'source':  ['Continuous', 'Discrete', 'Categorical']})
#worksheet.write_row('C2:C10', ['Integers', 2, 10])
worksheet.data_validation('C2:C20', {'validate': 'integer',
                                 'criteria': 'between',
                                 'minimum': 2,
                                 'maximum': 10})

#Freeze panes
worksheet.freeze_panes(1, 0)  # Freeze the first row

workbook.close()