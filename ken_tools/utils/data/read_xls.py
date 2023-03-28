import xlrd


class ReadExcel(object):
    def __init__(self, filename):
        super(ReadExcel, self).__init__()
        self.data = xlrd.open_workbook(filename)

    def open_a_sheet(self, sheet=0):
        if isinstance(sheet, int):
            table = self.data.sheet_by_index(sheet)
        elif isinstance(sheet, str):
            assert sheet in self.data.sheet_names()
            table = self.data.sheet_by_name(sheet)
        else:
            raise ValueError

        return table

    def read_a_row(self, sheet, rowx, start=0, end=None):
        table = self.open_a_sheet(sheet)
        assert rowx < table.nrows
        return table.row_values(rowx, start, end)

    def read_a_col(self, sheet, colx, start=0, end=None):
        table = self.open_a_sheet(sheet)
        assert colx < table.ncols
        return table.col_values(colx, start, end)

    def read_a_block(self, sheet, y1, y2, x1, x2):
        return [self.read_a_row(sheet, rowx, x1, x2+1) for rowx in range(y1, y2+1)]

    def read_one_sheet(self, sheet=0):
        table = self.open_a_sheet(sheet)
        nrows = table.nrows
        ncols = table.ncols
        return self.read_a_block(sheet, 0, nrows-1, 0, ncols-1)


if __name__ == "__main__":
    filename = 'data.xls'
    handle = ReadExcel(filename)
    from pprint import pprint
    pprint(handle.read_a_block(0, 0, 2, 1, 4))