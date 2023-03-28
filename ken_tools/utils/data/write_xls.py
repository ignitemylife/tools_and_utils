import os
import xlwt, xlrd
from xlutils.copy import copy


class WriteExcel(object):
    def __init__(self, file_name, sheet_name):
        self.file_name = file_name
        if os.path.exists(file_name):
            self.workbook = copy(xlrd.open_workbook(self.file_name, formatting_info=True))
        else:
            self.workbook = xlwt.Workbook()
        self.sheet = self.workbook.add_sheet(sheet_name, cell_overwrite_ok=True)
        self.row = 0

    def write_a_row(self, data_list):
        row_to_write = self.sheet.row(self.row)
        for index, data in enumerate(data_list):
            row_to_write.write(index, data)
        self.row += 1

    def write_a_cell(self, *args):
        raise NotImplementedError

    def save(self):
        self.workbook.save(self.file_name)
