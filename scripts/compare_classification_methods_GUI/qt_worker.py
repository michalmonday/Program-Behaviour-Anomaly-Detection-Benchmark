# from PyQt6.QtCore import QRunnable, pyqtSignal
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import sys

import traceback

class WorkerSignals(QObject):
    ''' Class copied from: https://www.pythonguis.com/tutorials/multithreading-pyqt6-applications-qthreadpool/
    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(tuple)


class Worker(QRunnable):
    ''' Class copied from: https://www.pythonguis.com/tutorials/multithreading-pyqt6-applications-qthreadpool/
    '''
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.kwargs['pyqt_progress_signal'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        ''' Initialise the runner function with passed args, kwargs.  '''
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

