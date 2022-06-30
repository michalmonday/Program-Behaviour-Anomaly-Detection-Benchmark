import os
from PyQt6 import QtGui, QtWidgets, QtCore
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from .file_load_status import FileLoadStatus
from .table_widget import TableWidget


class InputFilesTableWidget(TableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_file(self, f_name):
        row_index = self.rowCount()
        self.insertRow(row_index)
        basename = os.path.basename(f_name)
        with open(f_name) as f:
            line_count = len(f.readlines())
        self.setItem(row_index, 0, QTableWidgetItem(basename))
        self.setItem(row_index, 1, QTableWidgetItem(f_name)) # full file name with path
        self.setItem(row_index, 2, QTableWidgetItem(f'{line_count}') )
        self.setItem(row_index, 3, QTableWidgetItem(''))


    def update_file_status(self, args):
        # args = (file name, FileLoadStatus value, ...more args depending on status type)
        f_name, status, *more_args = args
        row_index = self.get_row_index_by_fname(f_name)
        status_item = self.item(row_index, 3)
        status = FileLoadStatus(status)
        if   status == FileLoadStatus.STARTED_LOADING:
            self.set_item_icon(status_item, 'SP_BrowserReload')
            status_item.setText('Loading...')
        elif status == FileLoadStatus.LOADED:
            status_item.setText('Loaded')
        elif status == FileLoadStatus.STARTED_GENERATING_ANOMALIES:
            status_item.setText('Generating anomalies...')
        elif status == FileLoadStatus.ANOMALIES_GENERATED:
            status_item.setText('Generated anomalies')
        elif status == FileLoadStatus.STARTED_REDUCING_LOOPS:
            status_item.setText('Reducing loops...')
        elif status == FileLoadStatus.REDUCED_LOOPS:
            status_item.setText('Reduced loops')
        elif status == FileLoadStatus.STARTED_GENERATING_WINDOWS:
            window_size = more_args[0]
            status_item.setText(f'Generating sliding windows of size {window_size}...')
        elif status == FileLoadStatus.WINDOWS_GENERATED:
            window_size = more_args[0]
            status_item.setText(f'Generated sliding windows of size {window_size}')
        elif status == FileLoadStatus.FINISHED:
            self.set_item_icon(status_item, 'SP_DialogApplyButton')
            self.set_row_color(row_index, QBrush(QColor.fromRgb(168, 255, 158)))
            status_item.setText('Done')

    # def set_file_started_loading(self, f_name):
    #     row_index = self.get_row_index_by_fname(f_name)
    #     self.set_item_icon(self.item(row_index, 3), 'SP_BrowserReload')

    # def set_file_loaded(self, f_name):
    #     row_index = self.get_row_index_by_fname(f_name)
    #     self.set_item_icon(self.item(row_index, 3), 'SP_DialogApplyButton')
    #     self.set_row_color(row_index, QBrush(QColor.fromRgb(168, 255, 158)))


    # def keyPressEvent(self, event):
    #     key = event.key()
    #     if key == Qt.Key_Return or key == Qt.Key_Enter:
    #         print('clicked enter')
    #     else:
    #         super(MyTableWidget, self).keyPressEvent(event)
