from PyQt6 import QtGui, QtWidgets, QtCore
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

class TableWidget(QTableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_row_index_by_fname(self, f_name):
        ''' fname = full file name '''
        for i in range(self.rowCount()):
            if self.item(i, 1).text() == f_name:
                return i
        raise Exception(f'The following file was not found in the table: "{f_name}"')

    def set_row_color(self, row_index, clr):
        for i in range(self.columnCount()):
            self.item(row_index, i).setBackground(clr)

    def set_item_icon(self, item, icon_name):
        ''' Icon names: https://www.pythonguis.com/faq/built-in-qicons-pyqt/ 
            For example:
            - SP_BrowserReload 
            - SP_DialogApplyButton   '''
        pixmapi = getattr(QStyle.StandardPixmap, icon_name)
        icon = self.style().standardIcon(pixmapi) 
        item.setIcon(icon)
        return icon

    def load_dataframe(self, df):
        ''' loads a pandas dataframe '''
        self.setRowCount(0)
        self.setColumnCount(0)

        column_names = ['Window size'] + df.columns.tolist()
        self.setColumnCount(len(column_names))
        self.setHorizontalHeaderLabels(column_names)

        for row_index, (df_index, row) in enumerate(df.iterrows()):
            print('\n\n\n')
            print(row_index, df_index, row)
            self.insertRow(row_index)
            # window size is the first item
            self.setItem(row_index, 0, QTableWidgetItem(str(df.index[row_index])))
            for col_index, (col_name, val) in enumerate(row.items()):
                print()
                print(col_index, col_name, val)
                self.setItem(row_index, col_index+1, QTableWidgetItem(str(val)))

        self.horizontalHeader().setStretchLastSection(True)

    def clear(self):
        self.setRowCount(0)

