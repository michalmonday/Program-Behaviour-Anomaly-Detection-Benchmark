import os
import sys
from PyQt6.QtCore import QRunnable, pyqtSignal, QObject
import traceback
import inspect
from .file_load_status import FileLoadStatus

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import compare_classification_methods_2 as ccm

class FileLoaderSignals(QObject):
    update_file_status = pyqtSignal(tuple) 
    # file_started_loading = pyqtSignal(str)
    # file_loaded = pyqtSignal(str) # emits fname
    # started_generating_anomalies = pyqtSignal(str) # fname
    # anomaly_generated = pyqtSignal(str, int) # str=fname, int=index of anomaly
    # sliding_windows_generated(str) # f_name
    finished = pyqtSignal(list, object) # all loaded fnames

class FileLoader(QRunnable):

    def __init__(self, f_names, main_window):#, load_and_process_kwargs, generate_artificial_kwargs, generate_sliding_windows_args):
        super().__init__()
        self.f_names = f_names
        self.signals = FileLoaderSignals()

        table = main_window.tableWidget_input_files
        self.load_and_process_kwargs = {
            'relative_pc' : main_window.checkBox_relative.isChecked(),
            'ignore_non_jumps' : main_window.checkBox_non_jumps.isChecked()
            }
        self.generate_artificial_kwargs = {
            'anomalies_per_normal_file' : int(main_window.lineEdit_anomalies_per_normal.text().strip()),
            'reduce_loops' : main_window.checkBox_reduce_loops.isChecked(),
            'reduce_loops_min_iteration_size' : int(main_window.lineEdit_minimum_loop_size.text().strip())
            }
        window_sizes = [int(i) for i in main_window.lineEdit_window_sizes.text().split(',')]
        self.generate_sliding_windows_args = [
            window_sizes,
            main_window.checkBox_append_features.isChecked(),
            ]
        self.signals.update_file_status.connect( table.update_file_status )

    def run(self):
        # possibly pass progress signal to emit
        ccm.load_and_preprocess_input_files( self.f_names, file_loader_signals=self.signals, **self.load_and_process_kwargs)

        ccm.generate_artificial_anomalies_from_training_dataset( file_loader_signals=self.signals, **self.generate_artificial_kwargs)


        df_stats = ccm.generate_sliding_windows(*self.generate_sliding_windows_args, file_loader_signals=self.signals)
        for f_name in self.f_names:
            self.signals.update_file_status.emit((f_name, FileLoadStatus.FINISHED.value))
        self.signals.finished.emit(self.f_names, df_stats)



