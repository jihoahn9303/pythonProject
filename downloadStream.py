import sys
from PyQt5.QtWidgets import *
# pip install PyQtWebEngine

from PyQt5.QtWebEngineWidgets import *
from PyQt5 import QtCore

from PyQt5.QtCore import pyqtSlot, pyqtSignal, QUrl
from lib.VideoDownloaderLayout import Ui_MainWindow

# User-define class : login class
from lib.AuthDialog import AuthDialog

import re
import datetime
import pymysql

# Youtebe download package
import pytube

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initAuthLock()
        self.initSignal()
        
        self.user_id = None
        self.user_pw = None
        
        self.is_play = False
        
        self.stream_list = None
        self.stream_size = 0
        
        self.youtube_obj = None

    # 접근 권한 제어(인증 전)
    def initAuthLock(self):
        self.priviewButton.setEnabled(False)
        self.fileNavButton.setEnabled(False)
        self.streamComboBox.setEnabled(False)
        self.startButton.setEnabled(False)
        self.urlTextEdit.setEnabled(False)
        self.pathTextEdit.setEnabled(False)
        self.showStatusMsg('인증 안됨')

    # 접근 권한 활성화(인증 후)
    def initAuthActive(self):
        self.priviewButton.setEnabled(True)
        self.fileNavButton.setEnabled(True)
        self.streamComboBox.setEnabled(True)
        self.urlTextEdit.setEnabled(True)
        self.pathTextEdit.setEnabled(True)
        self.showStatusMsg('인증 완료')

    def showStatusMsg(self, msg):
        self.statusbar.showMessage(msg)

    # 시그널 초기화 및 슬롯 설정
    def initSignal(self):
        self.exitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.loginButton.clicked.connect(self.authCheck)
        self.priviewButton.clicked.connect(self.load_url)
        self.webEngineView.loadProgress.connect(self.showProgressBrowserLoading)
        self.fileNavButton.clicked.connect(self.selectDownPath)
        self.startButton.clicked.connect(self.youtube_download)


    @pyqtSlot()
    def authCheck(self):
        dlg = AuthDialog()
        dlg.exec_()

        self.user_id = dlg.user_id
        self.user_pw = dlg.user_pw
        # 이 부분에서 필요한 경우 로컬 DB 또는 서버 연동 후
        # 유저 정보 및 사용 유효기간을 체크하는 코드를 삽입

        if True:
            self.initAuthActive()
            self.loginButton.setText("인증 완료")
            self.loginButton.setEnabled(False)
            self.urlTextEdit.setFocus()
            self.append_log_msg("Login Success")
        
        else:
            QMessageBox.about(self, "인증 오류", "아이디 또는 비밀번호를 확인하세요.")

    def dbConnect(self):
        dbconn = pymysql.connect(host = 'localhost', user = 'jihoahn', password = 'D6e72n9y!!', db = 'user_log', charset = 'utf8')
        return dbconn

    def append_log_msg(self, action):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        app_msg = self.user_id + " : " + action + ' - (' + now + ')'
        # print(app_msg)
        self.plainTextEdit.appendPlainText(app_msg) 

        conn = self.dbConnect()

        with conn:
            with conn.cursor() as c:
                c.execute('''
                    CREATE TABLE IF NOT EXISTS logs(
                        no INT AUTO_INCREMENT PRIMARY KEY,
                        id VARCHAR(25) NOT NULL,
                        log VARCHAR(100), date VARCHAR(25))
                ''')
                conn.commit()

                id = self.user_id
                log = app_msg

                c.execute("INSERT INTO logs(id, log, date) VALUES (%s,%s,%s)", (id, log, now))
                conn.commit()

    @pyqtSlot()
    def load_url(self):
        url = self.urlTextEdit.text().strip()
        v = re.compile('^https://www.youtube.com/?')

        # 중지 버튼을 클릭한 경우(동영상이 웹 엔진 뷰에 로딩 되었을 떄)
        if self.is_play:
            self.append_log_msg("Clicked the Stop Button")
            self.webEngineView.load(QUrl('about:blank'))
            self.priviewButton.setText("재생")
            self.is_play = False
            self.startButton.setEnabled(False)
            self.streamComboBox.clear()
            self.saveProgressBar.setValue(0)
            self.showStatusMsg("인증 완료")

        # 재생 버튼을 클릭한 경우(동영상이 웹 엔진 뷰에 없을 떄)
        else:
            if v.match(url) is not None:
                self.append_log_msg("Clicked the Play Button.")
                self.webEngineView.load(QUrl(url))
                self.showStatusMsg(url + "재생 중")
                self.priviewButton.setText("중지")
                self.is_play = True
                self.startButton.setEnabled(True)
                self.initStreamBox(url)

            else:
                QMessageBox.about(self, "오류", "Youtube 주소 형식이 아닙니다.")
                self.urlTextEdit.clear()
                self.urlTextEdit.setFocus(True)

    @pyqtSlot(int)
    def showProgressBrowserLoading(self, value):
        self.loadProgressBar.setValue(value)

    @pyqtSlot()
    def selectDownPath(self):
        # fname = QFileDialog.getOpenFileName(self)
        # self.pathTextEdit.setText(fname[0])

        fpath = QFileDialog.getExistingDirectory(self, 'Select directory')
        self.pathTextEdit.setText(fpath)

    def initStreamBox(self, url):
        self.youtube_obj = pytube.YouTube(url)
        self.stream_list = self.youtube_obj.streams.all()
        self.streamComboBox.clear()

        for element in self.stream_list:
            tmp_list = [str(element.mime_type), str(element.res), str(element.abr), str(element.fps)]
            true_list = [item for item in tmp_list if item is not None]
            self.streamComboBox.addItem(','.join(true_list))

    @pyqtSlot()
    def youtube_download(self):
        down_dir = self.pathTextEdit.text().strip()
        if down_dir == None or down_dir == '' or not down_dir:
            QMessageBox.about(self, '경로 선택', '다운 받을 경로를 선택하세요.')
            return None

        else:
            self.stream_size = self.stream_list[self.streamComboBox.currentIndex()].filesize
            self.stream_list[self.streamComboBox.currentIndex()].download(down_dir)
            self.append_log_msg("Download {}".format(self.youtube_obj.title))
            # Youtube interface callback function
            self.youtube_obj.register_on_progress_callback(self.youtubeDownProgress)


    # define callback function of the downloading progress
    def youtubeDownProgress(self, stream, chunk, file_handle, bytes_remaining):
        self.saveProgressBar.setValue(int(((self.stream_size - bytes_remaining) / self.stream_size) * 100))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    downloader = Main()
    downloader.show()
    app.exec_()
