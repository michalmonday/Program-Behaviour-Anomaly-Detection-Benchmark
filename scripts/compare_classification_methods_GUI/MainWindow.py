from .table_widget import TableWidget
from .line_edit import LineEdit
from .input_files_table_widget import InputFilesTableWidget
from .check_box import CheckBox
# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt6 UI code generator 6.3.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(881, 803)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setEnabled(True)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 861, 221))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.checkBox_relative = CheckBox(self.groupBox)
        self.checkBox_relative.setGeometry(QtCore.QRect(10, 30, 171, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_relative.setFont(font)
        self.checkBox_relative.setChecked(True)
        self.checkBox_relative.setObjectName("checkBox_relative")
        self.checkBox_non_jumps = CheckBox(self.groupBox)
        self.checkBox_non_jumps.setGeometry(QtCore.QRect(10, 50, 161, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_non_jumps.setFont(font)
        self.checkBox_non_jumps.setChecked(True)
        self.checkBox_non_jumps.setObjectName("checkBox_non_jumps")
        self.groupBox_artificial_anomalies = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_artificial_anomalies.setGeometry(QtCore.QRect(10, 80, 411, 101))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        self.groupBox_artificial_anomalies.setFont(font)
        self.groupBox_artificial_anomalies.setFlat(False)
        self.groupBox_artificial_anomalies.setCheckable(False)
        self.groupBox_artificial_anomalies.setObjectName("groupBox_artificial_anomalies")
        self.lineEdit_anomalies_per_normal = LineEdit(self.groupBox_artificial_anomalies)
        self.lineEdit_anomalies_per_normal.setGeometry(QtCore.QRect(10, 30, 31, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.lineEdit_anomalies_per_normal.setFont(font)
        self.lineEdit_anomalies_per_normal.setObjectName("lineEdit_anomalies_per_normal")
        self.label_anomalies_per_normal = QtWidgets.QLabel(self.groupBox_artificial_anomalies)
        self.label_anomalies_per_normal.setGeometry(QtCore.QRect(50, 30, 261, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        self.label_anomalies_per_normal.setFont(font)
        self.label_anomalies_per_normal.setObjectName("label_anomalies_per_normal")
        self.label_minimum_loop_size = QtWidgets.QLabel(self.groupBox_artificial_anomalies)
        self.label_minimum_loop_size.setEnabled(False)
        self.label_minimum_loop_size.setGeometry(QtCore.QRect(160, 60, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        self.label_minimum_loop_size.setFont(font)
        self.label_minimum_loop_size.setObjectName("label_minimum_loop_size")
        self.lineEdit_minimum_loop_size = LineEdit(self.groupBox_artificial_anomalies)
        self.lineEdit_minimum_loop_size.setEnabled(False)
        self.lineEdit_minimum_loop_size.setGeometry(QtCore.QRect(120, 60, 31, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.lineEdit_minimum_loop_size.setFont(font)
        self.lineEdit_minimum_loop_size.setObjectName("lineEdit_minimum_loop_size")
        self.checkBox_reduce_loops = CheckBox(self.groupBox_artificial_anomalies)
        self.checkBox_reduce_loops.setGeometry(QtCore.QRect(10, 60, 101, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_reduce_loops.setFont(font)
        self.checkBox_reduce_loops.setObjectName("checkBox_reduce_loops")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_3.setGeometry(QtCore.QRect(440, 80, 411, 101))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_window_sizes = QtWidgets.QLabel(self.groupBox_3)
        self.label_window_sizes.setGeometry(QtCore.QRect(180, 60, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        self.label_window_sizes.setFont(font)
        self.label_window_sizes.setObjectName("label_window_sizes")
        self.lineEdit_window_sizes = LineEdit(self.groupBox_3)
        self.lineEdit_window_sizes.setGeometry(QtCore.QRect(10, 60, 161, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.lineEdit_window_sizes.setFont(font)
        self.lineEdit_window_sizes.setObjectName("lineEdit_window_sizes")
        self.checkBox_append_features = CheckBox(self.groupBox_3)
        self.checkBox_append_features.setGeometry(QtCore.QRect(10, 30, 221, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_append_features.setFont(font)
        self.checkBox_append_features.setChecked(True)
        self.checkBox_append_features.setObjectName("checkBox_append_features")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setGeometry(QtCore.QRect(610, 500, 261, 121))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.checkBox_ngrams = CheckBox(self.groupBox_2)
        self.checkBox_ngrams.setGeometry(QtCore.QRect(10, 90, 121, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_ngrams.setFont(font)
        self.checkBox_ngrams.setChecked(True)
        self.checkBox_ngrams.setObjectName("checkBox_ngrams")
        self.checkBox_local_outlier_factor = CheckBox(self.groupBox_2)
        self.checkBox_local_outlier_factor.setGeometry(QtCore.QRect(10, 70, 131, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_local_outlier_factor.setFont(font)
        self.checkBox_local_outlier_factor.setChecked(True)
        self.checkBox_local_outlier_factor.setObjectName("checkBox_local_outlier_factor")
        self.checkBox_one_class_svm = CheckBox(self.groupBox_2)
        self.checkBox_one_class_svm.setGeometry(QtCore.QRect(10, 50, 121, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_one_class_svm.setFont(font)
        self.checkBox_one_class_svm.setChecked(True)
        self.checkBox_one_class_svm.setObjectName("checkBox_one_class_svm")
        self.checkBox_isolation_forest = CheckBox(self.groupBox_2)
        self.checkBox_isolation_forest.setGeometry(QtCore.QRect(10, 30, 121, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.checkBox_isolation_forest.setFont(font)
        self.checkBox_isolation_forest.setChecked(True)
        self.checkBox_isolation_forest.setObjectName("checkBox_isolation_forest")
        self.groupBox_input = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_input.setEnabled(True)
        self.groupBox_input.setGeometry(QtCore.QRect(10, 240, 861, 251))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.groupBox_input.setFont(font)
        self.groupBox_input.setObjectName("groupBox_input")
        self.tableWidget_input_files = InputFilesTableWidget(self.groupBox_input)
        self.tableWidget_input_files.setGeometry(QtCore.QRect(10, 40, 841, 201))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.tableWidget_input_files.setFont(font)
        self.tableWidget_input_files.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget_input_files.setObjectName("tableWidget_input_files")
        self.tableWidget_input_files.setColumnCount(4)
        self.tableWidget_input_files.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_input_files.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_input_files.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_input_files.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_input_files.setHorizontalHeaderItem(3, item)
        self.tableWidget_input_files.horizontalHeader().setDefaultSectionSize(108)
        self.tableWidget_input_files.horizontalHeader().setStretchLastSection(True)
        self.btn_load_and_preprocess_input_files = QtWidgets.QPushButton(self.centralwidget)
        self.btn_load_and_preprocess_input_files.setEnabled(True)
        self.btn_load_and_preprocess_input_files.setGeometry(QtCore.QRect(610, 700, 211, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.btn_load_and_preprocess_input_files.setFont(font)
        self.btn_load_and_preprocess_input_files.setObjectName("btn_load_and_preprocess_input_files")
        self.btn_train_test_evaluate = QtWidgets.QPushButton(self.centralwidget)
        self.btn_train_test_evaluate.setEnabled(True)
        self.btn_train_test_evaluate.setGeometry(QtCore.QRect(610, 730, 131, 24))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.btn_train_test_evaluate.setFont(font)
        self.btn_train_test_evaluate.setObjectName("btn_train_test_evaluate")
        self.groupBox_dataset = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_dataset.setGeometry(QtCore.QRect(10, 500, 591, 251))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.groupBox_dataset.setFont(font)
        self.groupBox_dataset.setObjectName("groupBox_dataset")
        self.tableWidget_dataset = TableWidget(self.groupBox_dataset)
        self.tableWidget_dataset.setGeometry(QtCore.QRect(10, 40, 571, 201))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.tableWidget_dataset.setFont(font)
        self.tableWidget_dataset.setObjectName("tableWidget_dataset")
        self.tableWidget_dataset.setColumnCount(0)
        self.tableWidget_dataset.setRowCount(0)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 881, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Anomaly detection in program behaviour benchmark tool"))
        self.groupBox.setTitle(_translate("MainWindow", "Preprocessing"))
        self.checkBox_relative.setText(_translate("MainWindow", "Relative program counters"))
        self.checkBox_non_jumps.setText(_translate("MainWindow", "Ignore non-jumps"))
        self.groupBox_artificial_anomalies.setTitle(_translate("MainWindow", "Artificial anomalies"))
        self.lineEdit_anomalies_per_normal.setText(_translate("MainWindow", "10"))
        self.label_anomalies_per_normal.setText(_translate("MainWindow", "anomalies per normal example * anomaly types"))
        self.label_minimum_loop_size.setText(_translate("MainWindow", "minimum loop size"))
        self.lineEdit_minimum_loop_size.setText(_translate("MainWindow", "10"))
        self.checkBox_reduce_loops.setText(_translate("MainWindow", "Reduce loops"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Sliding windows "))
        self.label_window_sizes.setText(_translate("MainWindow", "window sizes"))
        self.lineEdit_window_sizes.setText(_translate("MainWindow", "7,30,100"))
        self.lineEdit_window_sizes.setPlaceholderText(_translate("MainWindow", "7,30,100"))
        self.checkBox_append_features.setText(_translate("MainWindow", "Append features to sliding windows"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Detection methods"))
        self.checkBox_ngrams.setText(_translate("MainWindow", "N-grams"))
        self.checkBox_local_outlier_factor.setText(_translate("MainWindow", "Local outlier factor"))
        self.checkBox_one_class_svm.setText(_translate("MainWindow", "One class SVM"))
        self.checkBox_isolation_forest.setText(_translate("MainWindow", "Isolation forest"))
        self.groupBox_input.setTitle(_translate("MainWindow", "Input files"))
        item = self.tableWidget_input_files.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Basename"))
        item = self.tableWidget_input_files.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Full name"))
        item = self.tableWidget_input_files.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Line count"))
        item = self.tableWidget_input_files.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Status"))
        self.btn_load_and_preprocess_input_files.setText(_translate("MainWindow", "Load and preprocess input CSV files"))
        self.btn_train_test_evaluate.setText(_translate("MainWindow", "Train test evaluate"))
        self.groupBox_dataset.setTitle(_translate("MainWindow", "Dataset"))