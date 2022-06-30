from PyQt6.QtWidgets import QCheckBox

#https://www.riverbankcomputing.com/static/Docs/PyQt6/api/qtcore/qt.html?highlight=qt%20checkstate##CheckState
from PyQt6.QtCore import Qt
# Qt.CheckState.Checked

from .settings import settings

class CheckBox(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_persistent(self):
        key = f'{self.objectName()}/persistence'
        state = settings.value(key, None) 
        if state is not None:
            self.setCheckState(Qt.CheckState(state))
        self.stateChanged.connect(lambda state: settings.setValue(
            key, 
            state
        ))

