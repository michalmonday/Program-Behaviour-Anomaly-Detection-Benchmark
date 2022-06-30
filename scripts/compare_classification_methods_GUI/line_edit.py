from PyQt6.QtWidgets import QLineEdit

from .settings import settings

class LineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_persistent(self):
        ''' Unfortunately this can't be done at __init__ because of the way 
            pyuic generates .py file (it puts "setObjectName" after __init__
            so "objectName()" below can't work). '''
        text_key = f'{self.objectName()}/persistence'
        text = settings.value(text_key, None) 
        if text is not None:
            self.setText(text)

        self.editingFinished.connect(lambda : settings.setValue(
            text_key, 
            self.text()
        ))

        enabled_key = f'{self.objectName()}/enabled'
        enabled = settings.value(enabled_key, None)
        if enabled is not None:
            self.setEnabled(enabled.lower() == 'true')
        

