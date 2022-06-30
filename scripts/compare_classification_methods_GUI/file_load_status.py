''' file created to avoid circular imports '''
from enum import Enum

class FileLoadStatus(Enum):
    STARTED_LOADING = 1
    LOADED = 2
    STARTED_GENERATING_ANOMALIES = 3
    ANOMALIES_GENERATED = 4
    STARTED_REDUCING_LOOPS = 5
    REDUCED_LOOPS = 6
    STARTED_GENERATING_WINDOWS = 7
    WINDOWS_GENERATED = 8
    FINISHED = 9

