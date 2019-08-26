from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from PyQt5.QtMultimedia import QSound


class IntroWorker(QObject):
    startMsg = pyqtSignal(str, str)

    @pyqtSlot()
    def playBgm(self):
        self.intro = QSound('D:/AtomProject/python/section6/resource/intro.wav')
        self.intro.play()
        self.startMsg.emit("Anonymous", self.intro.fileName())
