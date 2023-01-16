import sys
import os
from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import traceback

from compare_classification_methods_GUI.qt_worker import Worker
from compare_classification_methods_GUI.MainWindow import Ui_MainWindow
from compare_classification_methods_GUI.settings import settings
from compare_classification_methods_GUI.file_loader import FileLoader

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 

import compare_classification_methods_2 as ccm
import utils


def make_persistent(app):
    ''' will loop over all widgets and call "make_persistent" if such 
    method is implemented. By default Qt widgets don't have such method,
    it is sometimes implemented by me in subclasses (if particular widget
    needs their state to be saved and restored) '''
    for w in app.allWidgets():
        func = getattr(w, "make_persistent", None)
        if callable(func):
            print(f'{w.objectName()} has make_persistent')
            w.make_persistent()

def save(settings, app):
    for w in app.allWidgets():
        mo = w.metaObject()
        if w.objectName() and not w.objectName().startswith("qt_"):
            settings.beginGroup(w.objectName())
            for i in range(mo.propertyCount()):
                prop = mo.property(i)
                name = prop.name()
                if prop.isWritable():
                    settings.setValue(name, w.property(name))
            settings.endGroup()


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, app, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.app = app
        # self.btn_download.clicked.connect(self.begin_download)
        self.btn_load_and_preprocess_input_files.clicked.connect(self.load_and_preprocess_input_files)
        self.btn_train_test_evaluate.clicked.connect(self.train_test_evaluate)
        self.btn_save_models.clicked.connect(self.save_models)
        self.btn_load_models.clicked.connect(self.load_models)

        self.btn_train_test_evaluate.setEnabled(False)
        self.btn_save_models.setEnabled(False)
        self.btn_load_models.setEnabled(False)
        # self.set_item_icon(self.btn_download, 'SP_ArrowDown')
        self.threadpool = QThreadPool()

        # self.lineEdit_anomalies_per_normal.make_persistent()
        # self.lineEdit_window_sizes.make_persistent()
        # self.checkBox_relative.make_persistent()
        make_persistent(self.app)

    def save_models(self):
        worker = Worker(ccm.save_models) 
        # worker.signals.progress.connect(self.on_train_test_evaluate_progress)
        # worker.signals.result.connect(self.on_train_test_evaluate_result)
        # worker.signals.finished.connect(lambda: self.statusBar().showMessage(''))
        self.threadpool.start(worker)

    def load_models(self):
        worker = Worker(ccm.load_models) 
        # worker.signals.progress.connect(self.on_train_test_evaluate_progress)
        # worker.signals.result.connect(self.on_train_test_evaluate_result)
        # worker.signals.finished.connect(lambda: self.statusBar().showMessage(''))
        self.threadpool.start(worker)

    # reduce loops change state behaviour
    # def on_reduce_loops_checkbox_changed(state):
    #     reduce_loops_enabled = state == Qt.CheckState.Checked.value
    #     self.lineEdit_minimum_loop_size.setEnabled(reduce_loops_enabled)
    #     self.label_minimum_loop_size.setEnabled(reduce_loops_enabled)
    #     self.checkBox_reduce_loops.stateChanged.connect(on_reduce_loops_checkbox_changed)

    #     # reduce loops initial state (unfortunately it must be done after UI_MainWindow.setupUI())
    #     reduce_loops_enabled = self.checkBox_reduce_loops.checkState() == Qt.CheckState.Checked 
    #     self.lineEdit_minimum_loop_size.setEnabled(reduce_loops_enabled)
    #     self.label_minimum_loop_size.setEnabled(reduce_loops_enabled) 


    def set_item_icon(self, item, icon_name):
        ''' Icon names: https://www.pythonguis.com/faq/built-in-qicons-pyqt/ 
            For example:
            - SP_BrowserReload 
            - SP_DialogApplyButton   '''
        pixmapi = getattr(QStyle.StandardPixmap, icon_name)
        icon = self.style().standardIcon(pixmapi) 
        item.setIcon(icon)
        return icon

    # def browse_output_dir(self):
    #     dir_ = QFileDialog.getExistingDirectory(self, "Select Directory")
    #     if dir_:
    #         dir_ = str(dir_)
    #         self.line_edit_output_dir.setText(dir_)
    #         settings.setValue('output_dir', dir_)
    #         self.output_dir = dir_

    def load_and_preprocess_input_files(self):
        self.btn_load_and_preprocess_input_files.setEnabled(False) 
        self.btn_train_test_evaluate.setEnabled(False)
        self.btn_save_models.setEnabled(False)
        self.btn_load_models.setEnabled(False)

        self.tableWidget_input_files.clear()
        files = QFileDialog.getOpenFileNames(self, "Select Directory")
        if not files or not files[0]:
            return 
        for f_name in files[0]:
            self.tableWidget_input_files.add_file(f_name)
        print(files[0])

        file_loader = FileLoader(files[0], self)
        file_loader.signals.finished.connect(self.on_file_load_finished)

        self.threadpool.start(file_loader)

    def on_file_load_finished(self, f_names, df_stats):
        self.btn_load_and_preprocess_input_files.setEnabled(True) 
        self.btn_train_test_evaluate.setEnabled(True)
        self.btn_save_models.setEnabled(True)
        self.btn_load_models.setEnabled(True)
        self.tableWidget_dataset.load_dataframe(df_stats)

    def train_test_evaluate(self):
        active_methods_map = {
            'N-grams'              : self.checkBox_ngrams.isChecked(),
            'Isolation forest'     : self.checkBox_isolation_forest.isChecked(),
            'One class SVM'        : self.checkBox_one_class_svm.isChecked(),
            'Local outlier factor' : self.checkBox_local_outlier_factor.isChecked()
        }
        # ccm.train_test_evaluate will automatically get supplied with "pyqt_progress_signal"
        # allowing to do:
        # pyqt_progress_signal.emit(('some string', 1, 'another string'))
        worker = Worker(ccm.train_test_evaluate, active_methods_map, dont_plot=True) 
        worker.signals.progress.connect(self.on_train_test_evaluate_progress)
        worker.signals.result.connect(self.on_train_test_evaluate_result)
        worker.signals.finished.connect(lambda: self.statusBar().showMessage(''))
        self.threadpool.start(worker)

    def on_train_test_evaluate_progress(self, progress_tuple):
        action, window_size, method_name, constructor_kwargs = progress_tuple
        print('train_test_evaluate progress:', action, window_size, method_name, constructor_kwargs)
        check_boxes = {
            'N-grams'              : self.checkBox_ngrams,
            'Isolation forest'     : self.checkBox_isolation_forest,
            'One class SVM'        : self.checkBox_one_class_svm,
            'Local outlier factor' : self.checkBox_local_outlier_factor
            }
        check_box = check_boxes[method_name]

        status_msg = f'{action} {method_name} window_size={window_size}, constructor_kwargs={constructor_kwargs}'
        self.statusBar().showMessage(status_msg)
        # check_box.setStyleSheet("QCheckBox::indicator"
        #                        "{"
        #                        "background-color : lightgreen;"
        #                        "}")

    def on_train_test_evaluate_result(self, df_results_all):
        utils.plot_results(df_results_all)



    def closeEvent(self, event):
        save(settings, self.app)
        super().closeEvent(event)


app = QtWidgets.QApplication(sys.argv)

window = MainWindow(app)
window.show()
app.exec()
