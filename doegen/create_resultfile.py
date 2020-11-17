"""
Generates Excel file with format for experimental design setup

Author: Sebastian Haan
Affiliation: Sydney Informatics Hub (SIH), THe University of Sydney
Version: 0.1
License: APGL-3.0
"""

import xlsxwriter

workbook = xlsxwriter.Workbook('Experiment_results_template.xlsx')
worksheet = workbook.add_worksheet()

workbook.set_properties({
    'title':    'Experimental Design Results',
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
worksheet.set_row(0, 20)

worksheet.set_column('A:H', 15, unlocked)

# Write the header cells
# Same identifier as in setupfile, this need to match with experimemnt setup file to merge with associated parameters!
heading1 = 'Nexp' 
# Optional: ID of measurement point (e.g. spatial or temporal position),
heading2 = 'PID' 
# Optional: index of multi output-target if applicable (optional)
#(e.g. Y can be distinct properties or target values even for same PID)
heading3 = 'Y Label' 
# Experiment or simulation result for given position PID and output Y Label
heading4 = 'Y Exp'
# Optional: ground truth for given PID and output Y Label
heading5 = 'Y Truth'
# Optional:  Standard deviation (noise) of experiment result for given position PID and output Y Label
heading6 = 'Std Y Exp'
# Optional:  Standard deviation (noise) of ground truth for given position PID and output Y Label
heading7 = 'Std Y Truth'
# Optional: weight for positional measurement with PID
heading8 = 'Weight PID'

worksheet.write('A1', heading1, header_format)
worksheet.write('B1', heading2, header_format)
worksheet.write('C1', heading3, header_format)
worksheet.write('D1', heading4, header_format)
worksheet.write('E1', heading5, header_format)
worksheet.write('F1', heading6, header_format)
worksheet.write('G1', heading7, header_format)
worksheet.write('H1', heading8, header_format)

#Freeze panes
worksheet.freeze_panes(1, 0)  # Freeze the first row

workbook.close()

print('Excel Template Created')