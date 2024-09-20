from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QSizePolicy, QFileDialog, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QLabel, QDialog
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QCamera
from PyQt5.QtCore import QUrl, Qt, QDir, QTimer
import sys
import json
import os


class CalVideo:
  def __init__(self, video_path) -> None:
    self.video_path = video_path
    self.video_name = video_path.replace("/", "\\").split(os.sep)[-1]
    self.l1_path = os.path.join(
      r"C:\Users\Tatocode\Documents\desk\dataset\final_video_eval\label", self.video_name.replace(".mp4", ".json"))
    self.l2_path = os.path.join(
      r"C:\Users\Tatocode\Documents\desk\dataset\final_video_eval\label2", self.video_name.replace(".mp4", ".json"))
    self.l1_cl, self.l1, self.l2 = self.parse_label(self.l1_path, self.l2_path)

  def diff(self, d1, d2):
    ks = d1.keys()
    for k in ks:
      if d1[k] != d2[k]:
        act = "take" if d1[k] > d2[k] else "replace"
        count = abs(d1[k] - d2[k])
        if k == "scissors":
          kind = "Surgical scissors"
        elif k == "forceps":
          kind = "Forceps"
        elif k == "gauze":
          kind = "Occlusive dressings"
        elif k == "kidney-dish":
          kind = "Kidney dish"
        break
    return act, kind, count

  def parse_label(self, l1_path, l2_path):
    with open(l1_path, "r") as rf:
      l1_ct = json.load(rf)
    with open(l2_path, "r") as rf:
      l2_ct = json.load(rf)
    ret1 = {}
    ret2 = {}
    before_state = {}
    for k in sorted(l1_ct):
      m, s = [int(i) for i in k.strip().split(":")]
      tm = m * 60 + s
      if tm == 0:
        before_state = l1_ct[k]
      else:
        act, name, count = self.diff(before_state, l1_ct[k])
        before_state = l1_ct[k]
        ret1[tm] = [act, name, count]
    for k in sorted(l2_ct):
      m, s = [int(i) for i in k.strip().split(":")]
      tm = m * 60 + s
      if l2_ct[k] == "left":
        pos = "left"
      elif l2_ct[k] == "middle":
        pos = "center"
      else:
        pos = "right"
      ret2[tm] = pos
    return l1_ct, ret1, ret2

  def get_info(self, current_time, info_box):
    info = []
    rules = {"Surgical scissor": "center", "Occlusive dressings": "right", "Forceps": "center", "Kidney dish": "center"}
    si_type = ""
    right_pos = ""
    for k in rules.keys():
      if info_box.toPlainText().endswith(k):
        right_pos = rules[k]
        si_type = k
        break
    if current_time in self.l1.keys():
      info.append(
        f"{self.l1[current_time][0]} {self.l1[current_time][2]} {self.l1[current_time][1]}")
    if current_time in self.l2.keys():
      if right_pos == "" or right_pos == self.l2[current_time]:
        info.append(f"{si_type} been placed at {self.l2[current_time]}, <span style='color:#00ff00'>Correct placement</span>")
      elif right_pos != "" and right_pos != self.l2[current_time]:
        info.append(f"{si_type} been placed at {self.l2[current_time]}, <span style='color:#ff0000'>Wrong placement</span>")
    return info
  
  def checklist(self, current_time):
    for k in sorted(self.l1_cl):
      m, s = [int(i) for i in k.strip().split(":")]
      tm = m * 60 + s
      if current_time == tm:
        return self.l1_cl[k]
    return ""


class MainWindow(QWidget):
  def __init__(self):
    super().__init__()

    self.setWindowTitle('Deep learning framework')
    self.setGeometry(500, 300, 1600, 1000)

    self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
    self.mediaPlayer.setMuted(True)
    self.mediaPlayer.stateChanged.connect(self.update_info_box)
    self.mediaPlayer.mediaStatusChanged.connect(self.check_media_status)

    self.camera = QCamera()

    self.videowidget = QVideoWidget()
    self.videowidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.videowidget.setStyleSheet("background-color:black;")

    self.openBtn = QPushButton('Upload Video')
    self.openBtn.clicked.connect(self.open_file)

    self.monitorBtn = QPushButton('Start monitoring')
    self.monitorBtn.clicked.connect(self.start_monitor)

    self.playBtn = QPushButton('Play / Pause')
    self.playBtn.clicked.connect(self.play_pause)
    self.playBtn.setEnabled(False)

    self.endBtn = QPushButton('Finish')
    self.endBtn.clicked.connect(self.end)
    self.endBtn.setEnabled(False)

    self.infoBox = QTextEdit()
    self.infoBox.setReadOnly(True)

    # 在按钮下方添加一个表格
    self.tableWidget = QTableWidget()
    self.tableWidget.setRowCount(2)  # 设置行数为2
    self.tableWidget.setColumnCount(2)  # 设置列数为2
    header = self.tableWidget.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Stretch)
    self.tableWidget.verticalHeader().setVisible(False)  # 隐藏行号
    self.tableWidget.horizontalHeader().setVisible(False)  # 隐藏列号
    sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
    self.tableWidget.setSizePolicy(sizePolicy)
    self.tableWidget.setMinimumHeight(0.26 * self.openBtn.height())
    self.tableWidget.setMaximumHeight(0.26 * self.openBtn.height())
    self.tableWidget.setSelectionMode(QAbstractItemView.NoSelection)
    

    self.rightLayout = QVBoxLayout()
    self.rightLayout.addWidget(self.openBtn)
    self.rightLayout.addWidget(self.monitorBtn)
    self.rightLayout.addWidget(self.playBtn)
    self.rightLayout.addWidget(self.endBtn)
    self.rightLayout.addWidget(self.tableWidget)  # 将表格添加到布局中
    self.rightLayout.addWidget(self.infoBox)

    self.layout = QHBoxLayout()
    self.layout.addWidget(self.videowidget, 5)
    self.layout.addLayout(self.rightLayout, 3)

    self.setLayout(self.layout)

    self.timer = QTimer(self)
    self.timer.timeout.connect(self.update_info_box)

    self.previousState = None
    self.isPaused = False
    self.isEnd = False

  def check_media_status(self, status):
    if status == QMediaPlayer.EndOfMedia:
      self.isEnd = True
      position = self.mediaPlayer.position() // 1000
      hours = position // 3600
      minutes = (position % 3600) // 60
      seconds = position % 60
      self.infoBox.append(f"<span style='color:#ff0000' ><b>{hours:02d}:{minutes:02d}:{seconds:02d} Video playback ends</b></span>")

  def open_file(self):
    self.camera.stop()
    self.camera.setViewfinder(None)
    fileName, _ = QFileDialog.getOpenFileName(
      self, "Open Movie", QDir.homePath())
    self.cal_video = CalVideo(fileName)
    self.before_cl = self.cal_video.checklist(0)
    fileName = fileName.replace("dataset/final_video_eval/video", "code/final_video_detection/result")
    if fileName != '':
      self.mediaPlayer.setVideoOutput(self.videowidget)
      self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(fileName)))
      self.mediaPlayer.play()
      self.openBtn.setEnabled(False)
      self.monitorBtn.setEnabled(False)
      self.playBtn.setEnabled(True)
      self.endBtn.setEnabled(True)
      self.timer.start(1000)

  def start_monitor(self):
    self.mediaPlayer.stop()
    self.mediaPlayer.setVideoOutput(None)
    self.layout.removeWidget(self.videowidget)
    self.videowidget.deleteLater()
    self.videowidget = QVideoWidget()
    self.videowidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.videowidget.setStyleSheet("background-color:black;")
    self.layout.insertWidget(0, self.videowidget, 5)
    self.camera.load()  # 加载摄像头
    self.camera.setViewfinder(self.videowidget)
    self.camera.start()
    self.openBtn.setEnabled(False)
    self.monitorBtn.setEnabled(False)
    self.playBtn.setEnabled(False)
    self.endBtn.setEnabled(True)

  def play_pause(self):
    if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
      self.timer.stop()
      self.mediaPlayer.pause()
    else:
      self.mediaPlayer.play()
      self.timer.start()

  def end(self):
    reply = QMessageBox.question(
      self, 'End', 'Are you sure you want to end it?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    if reply == QMessageBox.Yes:
      self.timer.destroyed()
      self.timer = QTimer(self)
      self.timer.timeout.connect(self.update_info_box)
      self.isPaused = False
      self.mediaPlayer.stop()
      self.mediaPlayer.setVideoOutput(None)
      self.camera.stop()
      self.camera.unload()
      self.camera.setViewfinder(None)
      self.infoBox.clear()
      self.openBtn.setEnabled(True)
      self.monitorBtn.setEnabled(True)
      self.playBtn.setEnabled(False)
      self.endBtn.setEnabled(False)

  def update_table(self, current_time):
    data = self.cal_video.checklist(current_time)
    if data != "":
      self.before_cl = data
    lst = []
    for k in sorted(self.before_cl):
      if k == "scissors":
          kind = "Surgical scissors"
      elif k == "forceps":
        kind = "Forceps"
      elif k == "gauze":
        kind = "Occlusive dressings"
      elif k == "kidney-dish":
        kind = "Kidney dish"
      lst.append(f"{kind}: {self.before_cl[k]}")
    data = [[lst[0], lst[1]], [lst[2], lst[3]]]
    # 将内容添加到表格中
    for i in range(len(data)):
      for j in range(len(data[i])):
        self.tableWidget.setItem(i, j, QTableWidgetItem(data[i][j]))

  def update_info_box(self):
    position = self.mediaPlayer.position() // 1000
    hours = position // 3600
    minutes = (position % 3600) // 60
    seconds = position % 60
    info = self.cal_video.get_info(60 * minutes + seconds, self.infoBox)
    self.update_table(minutes * 60 + seconds)
    if len(info) != 0:
      for io in info:
        self.infoBox.append(f"<b>{hours:02d}:{minutes:02d}:{seconds:02d}</b> {io}")
    if self.mediaPlayer.state() == QMediaPlayer.PlayingState and self.previousState != QMediaPlayer.PlayingState and not self.isEnd:
      self.infoBox.append(f"<b>{hours:02d}:{minutes:02d}:{seconds:02d} Video is playing</b>")
      self.isPaused = False
    elif self.mediaPlayer.state() != QMediaPlayer.PlayingState and self.previousState != QMediaPlayer.PlayingState and not self.isPaused and not self.isEnd:
      self.infoBox.append(f"<b>{hours:02d}:{minutes:02d}:{seconds:02d} Video paused</b>")
      self.isPaused = True
    self.previousState = self.mediaPlayer.state()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
