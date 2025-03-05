# -*- coding: utf-8 -*-
import io
import os
import json
import queue
import random
import re
import sys
import threading
import time
from base64 import b64encode
from queue import Queue
from system_hotkey import SystemHotkey
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
import jieba
import numpy as np
import typing_extensions
import win32api
import win32con
from skimage.metrics import structural_similarity as ssim
import pyautogui
from cv2 import cvtColor, COLOR_BGR2GRAY, calcHist, resize
from PIL import Image
from numpy import uint8,frombuffer
import subprocess  # 进程，管道
from base64 import b64encode
from json import loads as jsonLoads, dumps as jsonDumps
from sys import platform as sysPlatform  # popen静默模式
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QThread, QPoint, QRect, QEvent, QRectF
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QTextOption, QFont, QTextCharFormat, QColor, QPen, QCursor, QBitmap, QPainter, \
    QBrush, QMouseEvent, QGuiApplication, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, \
    QTextEdit, QSplitter, QCheckBox, QComboBox, QAction, QMenu, QTextBrowser, QDesktopWidget, QFileDialog, QMessageBox, \
    QStatusBar
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QLabel, QSpinBox, \
            QPushButton, QCheckBox, QDialogButtonBox
#COR引擎
class OcrApi():
    def __init__(self, exePath: str, argument: dict = None):
        """初始化识别器。\n
        `exePath`: 识别器`PaddleOCR_json.exe`的路径。\n
        `argument`: 启动参数，字典`{"键":值}`。
            指定不同语言的配置文件路径，识别多国语言
            {'config_path':"./PaddleOCR-json/models/config_chinese.txt"}
            启用cls方向分类，识别方向不是正朝上的图片。默认false
            {'cls':false}
            启用方向分类，必须与cls值相同。
            {'use_angle_cls':false}
            启用CPU推理加速，关掉可以减少内存占用，但会降低速度。默认true
            {'enable_mkldnn':true}
            对图像边长进行限制，降低分辨率，加快速度。
            {'limit_side_len':960}
        """
        cwd = os.path.abspath(os.path.join(exePath, os.pardir))  # 获取exe父文件夹
        # 处理启动参数
        if not argument == None:
            for key, value in argument.items():
                if isinstance(value, str):  # 字符串类型的值加双引号
                    exePath += f' --{key}="{value}"'
                else:
                    exePath += f" --{key}={value}"
        # 设置子进程启用静默模式，不显示控制台窗口
        startupinfo = None
        if "win32" in str(sysPlatform).lower():
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags = subprocess.CREATE_NEW_CONSOLE | subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        self.ret = subprocess.Popen(  # 打开管道
            exePath, cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # 丢弃stderr的内容
            startupinfo=startupinfo  # 开启静默模式
        )
        # 启动子进程
        while True:
            if not self.ret.poll() == None:  # 子进程已退出，初始化失败
                raise Exception(f"OCR init fail.")
            initStr = self.ret.stdout.readline().decode("utf-8", errors="ignore")
            if "OCR init completed." in initStr:  # 初始化成功
                break
    def run_dict(self, writeDict: dict):
        """传入指令字典，发送给引擎进程。\n
        `writeDict`: 指令字典。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        # 检查子进程
        if not self.ret.poll() == None:
            return {"code": 901, "data": f"子进程已崩溃。"}
        # 输入信息
        writeStr = jsonDumps(writeDict, ensure_ascii=True, indent=None)+"\n"
        try:
            self.ret.stdin.write(writeStr.encode("utf-8"))
            self.ret.stdin.flush()
        except Exception as e:
            return {"code": 902, "data": f"向识别器进程传入指令失败，疑似子进程已崩溃。{e}"}
        # 获取返回值
        try:
            getStr = self.ret.stdout.readline().decode("utf-8", errors="ignore")
        except Exception as e:
            return {"code": 903, "data": f"读取识别器进程输出值失败。异常信息：[{e}]"}
        try:
            return jsonLoads(getStr)
        except Exception as e:
            return {"code": 904, "data": f"识别器输出值反序列化JSON失败。异常信息：[{e}]。原始内容：[{getStr}]"}
    def run(self, imgPath: str):
        """对一张本地图片进行文字识别。\n
        `exePath`: 图片路径。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        writeDict = {"image_path": imgPath}
        return self.run_dict(writeDict)
    def runClipboard(self):
        """立刻对剪贴板第一位的图片进行文字识别。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        return self.run("clipboard")
    def runBase64(self, imageBase64: str):
        """对一张编码为base64字符串的图片进行文字识别。\n
        `imageBase64`: 图片base64字符串。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        writeDict = {"image_base64": imageBase64}
        return self.run_dict(writeDict)
    def runBytes(self, imageBytes):
        """对一张图片的字节流信息进行文字识别。\n
        `imageBytes`: 图片字节流。\n
        `return`:  {"code": 识别码, "data": 内容列表或错误信息字符串}\n"""
        imageBase64 = b64encode(imageBytes).decode('utf-8')
        return self.runBase64(imageBase64)
    def exit(self):
        """关闭引擎子进程"""
        self.ret.kill()  # 关闭子进程
    def __del__(self):
        self.exit()
#OCR服务线程
class OcrThread(QThread):
    ocr_result_signal=pyqtSignal(str)
    ocr_run_signal = True
    #初始化
    def __init__(self):
        super().__init__()
        self.config=ConfigManager()
        self.exepath=self.config.get_item("OCR路径")
        self.task_queue = queue.Queue()
        self.ocr = OcrApi(self.exepath)
    # 转码
    def pixmap_to_array(self, pixmap, channels_count=4):
        size = pixmap.size()
        width = size.width()
        height = size.height()
        # print(size,width,height)
        image = pixmap.toImage()
        s = image.bits().asstring(width * height * channels_count)
        # img = fromstring(s, dtype=uint8).reshape((height, width, channels_count))
        img = frombuffer(s, dtype=uint8).reshape((height, width, channels_count))
        img = img[:, :, :3]
        # ("1111 : {}".format(type(img)))
        return img.astype(uint8)
    #图片装bytes
    def array_to_pixmap(self, arr):
        # 将numpy数组转换为字节流
        # 将numpy数组转换为PIL图像
        img = Image.fromarray(np.uint8(arr))
        # 创建一个BytesIO对象
        byte_io = io.BytesIO()
        # 保存图像到BytesIO对象
        img.save(byte_io, 'PNG')
        # 获取字节流
        byte_io.seek(0)
        byte_data = byte_io.read()
        return byte_data
    def run(self):
        while self.ocr_run_signal:
            if not self.task_queue.empty():
                image = self.task_queue.get()
                arrayimage1=self.pixmap_to_array(image)
                arrayimage2=self.array_to_pixmap(arrayimage1)
                if type(image)==QPixmap:
                    res = self.ocr.runBytes(arrayimage2)
                    if res["code"] == 100:
                        newres= ""
                        for line in res["data"]:
                            newres += line["text"] + "\n"
                    else:
                        newres="错误！未识别到文字！"
                    self.ocr_result_signal.emit(str(newres))
            else:
                time.sleep(1)
        self.quitocr()
    def quitocr(self):
        self.ocr.exit()
        self.ocr_run_signal=False
        print("OCR服务已关闭！")
#翻译服务线程
class TranThread(QThread):
    tran_result_signal=pyqtSignal(str)
    tran_run_signal = True
    #初始化模型
    def __init__(self):
        super().__init__()
        self.task_queue = queue.Queue()
        # 中文模型
        self.model_path = 'models/translate/zh-en'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.pipeline2 = transformers.pipeline("translation", model=self.translate_model, tokenizer=self.tokenizer)
        #英文模型
        self.model_path2 = 'models/translate/en-zh'
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.model_path2)
        self.translate_model2 = AutoModelForSeq2SeqLM.from_pretrained(self.model_path2)
        self.pipeline = transformers.pipeline("translation", model=self.translate_model2, tokenizer=self.tokenizer2)
    #替换连续的相同内容
    def process_word(self, text):
        if re.search(r'[\u4e00-\u9fff]', text):
            text = text.replace(' ', '')
            if len(text) > 1:
                result = re.sub(r'([\u4e00-\u9fff])\1+', r'\1', text)  # 替换连续的相同中文字为第一个中文字
                return result
            else:
                return text
        else:
            result = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)  # 替换连续的相同字母为第一个字母
            return result
    # 匹配最后一个标点符号的正则表达式
    def remove_last_punctuation(self, text):
        pattern = r'[,.?;]$'
        punctuation_list = []
        cleaned_list = []
        for sentence in text:
            match = re.search(pattern, sentence)
            if match:
                # 如果匹配到了标点符号，则去除标点符号并记录到结果列表中
                cleaned_sentence = re.sub(pattern, '', sentence)
                punctuation = match.group()
            else:
                # 如果没有匹配到标点符号，则保持原始文本并将结果列表中添加空字符串
                cleaned_sentence = sentence
                punctuation = ''
            punctuation_list.append(punctuation)
            cleaned_list.append(cleaned_sentence)
        return punctuation_list, cleaned_list
    # 计算similarity
    def cosine_similarity(self, u, v):
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        similarity = dot_product / (norm_u * norm_v)
        return similarity
    # 词语判定
    def do_sentence(self, translated_text):
        if translated_text == '':
            return ''
        translated_text = self.process_word(translated_text)
        # (translated_text)
        tokenized_sentence = list(jieba.cut(translated_text))
        # 保留第一个词语
        words = [tokenized_sentence[0]]  # 保留第一个词语
        for i in range(1, len(tokenized_sentence)):
            if tokenized_sentence[i] != tokenized_sentence[i - 1]:
                words.append(tokenized_sentence[i])
        words = [self.process_word(word) for word in words]
        # print(words)
        filtered_sentence = "".join(words)
        sentences = []
        j = ''
        for i in words:
            j = j + i
            if i[-1] in [',', '?', '!', ')', ';', ' ', '...']:
                sentences.append(j)
                j = ''
        sentences.append(j)
        if len(sentences) <= 1:
            return filtered_sentence
        # 分词和构建词袋表示
        punctuation_list, sentences = self.remove_last_punctuation(sentences)
        # print(punctuation_list)
        # print(sentences)
        tokenized_sentences = [list(jieba.cut(sentence)) for sentence in sentences]
        vocabulary = set()
        for sentence in tokenized_sentences:
            vocabulary.update(sentence)
        vocabulary = list(vocabulary)
        word_to_index = {word: i for i, word in enumerate(vocabulary)}
        bag_of_words = np.zeros((len(sentences), len(vocabulary)))
        for i, sentence in enumerate(tokenized_sentences):
            for word in sentence:
                word_index = word_to_index[word]
                bag_of_words[i, word_index] += 1

        # 计算词频分布
        word_frequency = bag_of_words / np.linalg.norm(bag_of_words, axis=1, keepdims=True)

        # 保留第一个句子
        filtered_sentences = [sentences[0] + punctuation_list[0]]  # 保留第一个句子
        for i in range(1, len(sentences)):
            similarity = self.cosine_similarity(word_frequency[i - 1], word_frequency[i])
            # print(similarity)
            if similarity < 0.75:
                filtered_sentences.append(sentences[i] + punctuation_list[i])
                # print(punctuation_list[i])
        # 输出结果
        sentence = ''.join(filtered_sentences)
        return sentence
    # 英译中
    def translate_entoch(self, text):
        translate_text = self.pipeline(text)[0]['translation_text']
        translate_text = self.do_sentence(translate_text)
        return translate_text
    # 中译英
    def translate_chtoen(self, text):
        translate_text = self.pipeline2(text)[0]['translation_text']
        translate_text = self.do_sentence(translate_text)
        return translate_text
    #服务入口
    def run(self):
        while self.tran_run_signal:
            if not self.task_queue.empty():
                text = self.task_queue.get()
                log = False
                for _char in text:
                    if '\u4e00' <= _char <= '\u9fa5':
                        log = True
                        break
                if log == True:
                    respon = self.translate_chtoen(text)  # 中文译英文
                elif log == False:
                    respon = self.translate_entoch(text)  # 英文译中文
                else:
                    respon = "无结果"
                self.tran_result_signal.emit(respon)
            else:
                time.sleep(1)
        print("TRAN服务已关闭！")
#程序配置文件
class ConfigManager:
    #初始化，查看是否有config，如果没有则创建
    def __init__(self, filename="./config.json"):
        self.file_path = filename
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w',encoding="utf-8") as file:
                config_first_data={
                    "识别结果复制到剪贴板":True,
                    "字幕字体大小":14,
                    "连续识别间隔":0.3,
                    "截图快捷键1":"alt",
                    "截图快捷键2": "q",
                    "识别快捷键1": "alt",
                    "识别快捷键2": "w",
                    "字幕快捷键1": "alt",
                    "字幕快捷键2": "e",
                    "OCR路径": "./PaddleOCR-json/PaddleOCR-json.exe",
                    "OCR引擎":False,
                    "翻译引擎":False,
                    "识别后翻译": False,
                    "显示字幕":False,
                    "字幕窗口X":0,
                    "字幕窗口Y":0,
                    "字幕窗口W":200,
                    "字幕窗口H":200,
                    "字幕描边颜色":"#FF00FF",
                    "字幕描边宽度":0.6,
                    "字幕透明度":0.4,
                    "窗口字体大小":12,
                    "连续识别相似度":0.95}
                json.dump(config_first_data, file, indent=4)
    #加载配置文件
    def load_config(self):
        with open(self.file_path, 'r',encoding="utf-8") as file:
            return json.load(file)
    #保存配置到文件
    def save_config(self, config_data):
        with open(self.file_path, 'w',encoding="utf-8") as file:
            json.dump(config_data, file, indent=4)
    #添加配置项
    def add_item(self, key, value):
        config_data = self.load_config()
        config_data[key] = value
        self.save_config(config_data)
    #删除配置项
    def delete_item(self, key):
        config_data = self.load_config()
        if key in config_data:
            del config_data[key]
            self.save_config(config_data)
    #查询配置项
    def get_item(self, key):
        config_data = self.load_config()
        return config_data.get(key, None)
    #修改配置项
    def update_item(self, key, new_value):
        config_data = self.load_config()
        if key in config_data:
            config_data[key] = new_value
            self.save_config(config_data)
    '''config_manager = ConfigManager()
    # 添加配置项
    config_manager.add_item("测试项目1", "example_value1")
    # 查询配置项
    pisdata=config_manager.get_item("测试项目1")
    # 修改配置项
    config_manager.update_item("测试项目1", "new_example_value1")
    # 删除配置项
    config_manager.delete_item("测试项目2")
    #获取全部数据
    alldata=config_manager._load_config()'''
#截屏窗口
class Screenshot(QWidget):
    continuousRecognizelog = pyqtSignal(bool)
    screenshotTaken = pyqtSignal(QPixmap)
    fullScreenImage = None # 初始化变量
    captureImage = None
    isMousePressLeft = None
    beginPosition = None
    endPosition = None
    painter = QPainter()# 创建 QPainter 对象
    #初始化
    def __init__(self,mode=1):
        super().__init__()
        self.mode=mode
        self.initWindow()  # 初始化窗口
        self.captureFullScreen()  # 捕获全屏
    #初始化窗口
    def initWindow(self):
        self.setCursor(Qt.CrossCursor)  # 设置光标
        self.setWindowFlag(Qt.FramelessWindowHint)  # 产生无边框窗口，用户不能通过窗口系统移动或调整无边界窗口的大小
        self.setWindowState(Qt.WindowFullScreen)  # 窗口全屏无边框
    #捕获全屏
    def captureFullScreen(self):
        # 捕获当前屏幕，返回像素图
        self.fullScreenImage = QGuiApplication.primaryScreen().grabWindow(QApplication.desktop().winId())
    #鼠标按下事件
    def mousePressEvent(self, event):
        # 如果鼠标事件为左键，则记录起始鼠标光标相对于窗口的位置
        if event.button() == Qt.LeftButton:
            self.beginPosition = event.pos()
            self.isMousePressLeft = True
        # 如果鼠标事件为右键，如果已经截图了则重新开始截图，如果没有截图就退出
        if event.button() == Qt.RightButton:
            if self.captureImage is not None:
                self.captureImage = None
                self.update()  # 更新，会擦除之前的选框
            else:
                self.close()
    #鼠标移动事件
    def mouseMoveEvent(self, event):
        if self.isMousePressLeft is True:
            self.endPosition = event.pos()
            self.update()
    #鼠标释放事件
    def mouseReleaseEvent(self, event):
        self.endPosition = event.pos()
        self.isMousePressLeft = False
        if self.mode==1:
            if self.captureImage is not None:
                self.screenshotTaken.emit(self.captureImage)
                self.close()
            else:
                self.screenshotTaken.emit(self.fullScreenImage)
                self.close()
        elif self.mode==2:
            self.saveImage()
        elif self.mode==3:
            self.get_coordinate()
        else:
            pass
    #绘制背景图
    def paintBackgroundImage(self):
        # 填充颜色，黑色半透明
        fillColor = QColor(0, 0, 0, 100)
        # 加载显示捕获的图片到窗口
        self.painter.drawPixmap(0, 0, self.fullScreenImage)
        # 填充颜色到给定的矩形
        self.painter.fillRect(self.fullScreenImage.rect(), fillColor)
    #获取矩形选框
    def getRectangle(self, beginPoint, endPoint):
        # 计算矩形宽和高
        rectWidth = int(abs(beginPoint.x() - endPoint.x()))
        rectHeight = int(abs(beginPoint.y() - endPoint.y()))
        # 计算矩形左上角 x 和 y
        rectTopleftX = beginPoint.x() if beginPoint.x() < endPoint.x() else endPoint.x()
        rectTopleftY = beginPoint.y() if beginPoint.y() < endPoint.y() else endPoint.y()
        # 构造一个以（x，y）为左上角，给定宽度和高度的矩形
        pickRect = QRect(rectTopleftX, rectTopleftY, rectWidth, rectHeight)
        # logging.info('开始坐标：%s,%s', beginPoint.x(),beginPoint.y())
        # logging.info('结束坐标：%s,%s', endPoint.x(), endPoint.y())
        return pickRect
    #绘制选框
    def paintSelectBox(self):
        penColor = QColor(255, 0, 0)  #画笔颜色
        self.painter.setPen(QPen(penColor, 2, Qt.SolidLine))# 设置画笔属性，蓝色、2px大小、实线
        if self.isMousePressLeft is True:
            pickRect = self.getRectangle(self.beginPosition, self.endPosition)  # 获得要截图的矩形框
            self.captureImage = self.fullScreenImage.copy(pickRect)  # 捕获截图矩形框内的图片
            self.painter.drawPixmap(pickRect.topLeft(), self.captureImage)  # 填充截图的图片
            self.painter.drawRect(pickRect)  # 绘制矩形边框
    #接收绘制事件开始绘制
    def paintEvent(self, event):
        self.painter.begin(self)  # 开始绘制
        self.paintBackgroundImage()  # 绘制背景
        self.paintSelectBox()  # 绘制选框
        self.painter.end()  # 结束绘制
    #保存图片
    def saveImage(self):
        random1=random.randint(0,9)
        random2=random.randint(0,9)
        fileName = QFileDialog.getSaveFileName(self, '保存图片', time.strftime("%Y年%m月%d日%H时%M分%S秒截图")+str(random1)+str(random2),".png")# 获取用户选择的文件名的完整路径
        if self.captureImage is not None: # 保存用户选择的文件。如果选取了区域，就保存区域图片；如果没有选取区域，就保存全屏图片
            self.captureImage.save(fileName[0] + fileName[1])
        else:
            self.fullScreenImage.save(fileName[0] + fileName[1])
        self.close()
    def get_coordinate(self):
        # 计算矩形宽和高
        zuobiao.prtW= int(abs(self.beginPosition.x() - self.endPosition.x()))
        zuobiao.prtH = int(abs(self.beginPosition.y() - self.endPosition.y()))
        # 计算矩形左上角 x 和 y
        zuobiao.prtX = int(self.beginPosition.x()) if self.beginPosition.x() < self.endPosition.x() else int(self.endPosition.x())
        zuobiao.prtY = int(self.beginPosition.y()) if self.beginPosition.y() < self.endPosition.y() else int(self.endPosition.y())
        self.close()
        self.continuousRecognizelog.emit(True)
    #ESC退出
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
#常驻识别框窗口
class ContinuousRecognizeWindows(QMainWindow):
    def __init__(self, X, Y, W, H):
        try:
            super(ContinuousRecognizeWindows, self).__init__()
            self.setGeometry(X, Y, W, H)
            # 窗口无标题栏、窗口置顶、窗口透明
            self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.Label = QLabel(self)
            self.Label.setObjectName("dragLabel")
            self.Label.setGeometry(0, 0, W, H)
            self.Label.setStyleSheet("border-width:1;\
                                  border:2px dashed #1E90FF;\
                                  background-color:rgba(62, 62, 62, 0.01)")

            # 此Label用于当鼠标进入界面时给出颜色反应
            self.dragLabel = QLabel(self)
            self.dragLabel.setObjectName("dragLabel")
            self.dragLabel.setGeometry(0, 0, 4000, 2000)

            self.Font = QFont()
            self.Font.setFamily("华康方圆体W7")
            self.Font.setPointSize(12)
            self.Button=11
            # 右下角用于拉伸界面的控件
            self.statusbar = QStatusBar(self)
            self.setStatusBar(self.statusbar)
            self._startPos = None
        except Exception as ex:
            print("错误信息："+str(ex))
    # 鼠标移动事件
    def mouseMoveEvent(self, e: QMouseEvent):
        try:
            if self._startPos is not None:
                self._endPos = e.pos() - self._startPos
                self.move(self.pos() + self._endPos)
        except Exception as ex:
            print("错误信息："+str(ex))
    # 鼠标按下事件
    def mousePressEvent(self, e: QMouseEvent):
        try:
            if e.button() == Qt.LeftButton:
                self._isTracking = True
                self._startPos = QPoint(e.x(), e.y())
        except Exception as ex:
            print("错误信息："+str(ex))
    # 鼠标松开事件
    def mouseReleaseEvent(self, e: QMouseEvent):
        try:
            rect = self.geometry()
            X1 = rect.left()
            Y1 = rect.top()
            X2 = rect.left() + rect.width()
            Y2 = rect.top() + rect.height()
            zuobiao.prtX=X1
            zuobiao.prtY=Y1
            zuobiao.prtW=X2-X1
            zuobiao.prtH=Y2-Y1
            if e.button() == Qt.LeftButton:
                self._isTracking = False
                self._startPos = None
                self._endPos = None
        except Exception as ex:
            print("错误信息："+str(ex))
    # 鼠标进入控件事件
    def enterEvent(self, QEvent):
        try:
            rect = self.geometry()
            X1 = rect.left()
            Y1 = rect.top()
            X2 = rect.left() + rect.width()
            Y2 = rect.top() + rect.height()
            self.dragLabel.setStyleSheet("background-color:rgba(62, 62, 62, 0.3)")
        except Exception as ex:
            print("错误信息："+str(ex))
    # 鼠标离开控件事件
    def leaveEvent(self, QEvent):
        try:
            self.dragLabel.setStyleSheet("background-color:none")
            self.Label.setGeometry(0, 0, self.width(), self.height())
            rect = self.geometry()
            X1 = rect.left()
            Y1 = rect.top()
            X2 = rect.left() + rect.width()
            Y2 = rect.top() + rect.height()
            zuobiao.prtX = X1
            zuobiao.prtY = Y1
            zuobiao.prtW = X2 - X1
            zuobiao.prtH = Y2 - Y1
        except Exception as ex:
            print("错误信息："+str(ex))
#储存坐标的类
class ContinuousRecognize_xy():
    def __init__(self):
        #self.screen = self.primaryScreen()  # 获取主屏幕
        #self.screen_geometry = self.screen.geometry()  # 获取屏幕的几何尺寸
        #self.width = self.screen_geometry.width()  # 屏幕宽度
        #self.height = self.screen_geometry.height()  # 屏幕高度
        #self.prtX = int(self.width / 2 - 200)
        #self.prtY = int(self.height * 4 / 5)
        self.prtX = 300
        self.prtY = 300
        self.prtW = 400
        self.prtH = 50
        self.overlog=False
#字幕窗口
class Subtitleswindows(QMainWindow):
    def __init__(self):
        super(Subtitleswindows, self).__init__()
        self.config=ConfigManager()
        self.data=self.config.load_config()
        self.setupui()
    def setupui(self):
        self.is_dragging = False  # 移动窗口开关
        self.drag_position = None  #移动窗口初始点位
        self.sezelog = False  #调整窗口大小开关
        self.stars=None  #调整大小初始参考初始点
        self.setObjectName("字幕")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint|Qt.SplashScreen)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background:transparent;")
        #self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
        #self.setAttribute(Qt.WA_TranslucentBackground)
        #self.setWindowOpacity(0.1)#窗口整体透明
        self.textbox = QTextBrowser(self)
        self.textbox.setWordWrapMode(QTextOption.WordWrap)#自动换行
        self.ku = self.data.get("字幕窗口W")
        self.gao =self.data.get("字幕窗口H")
        self.xxs =self.data.get("字幕窗口X")
        self.yys =self.data.get("字幕窗口Y")
        self.move(self.xxs, self.yys)
        self.resize(1900,1050)
        self.textbox.resize(self.ku,self.gao)
        self.textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.textbox.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        transparency=self.data.get("字幕透明度")
        self.textbox.setStyleSheet("border-width:0;border-style:outset;border-top:0px solid #00FF00;color:white;font-weight: bold;background-color:rgba(0, 0, 0, {})".format(transparency))
        font = QFont()
        font.setFamily("微软雅黑")
        font_sizes=self.data.get("字幕字体大小")
        font.setPointSize(font_sizes)
        self.textbox.setFont(font)
        self.format = QTextCharFormat()
        fontcolor=self.data.get("字幕描边颜色")
        fontOutline=self.data.get("字幕描边宽度")
        self.format.setTextOutline(QPen(QColor(fontcolor), fontOutline, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        self.textbox.mergeCurrentCharFormat(self.format)
        self.textbox.mousePressEvent = self.mousePressEvent
        self.textbox.mouseMoveEvent = self.mouseMoveEvent
        self.textbox.mouseReleaseEvent = self.mouseReleaseEvent
        self.textbox.contextMenuEvent = self.contextMenuEvent
        self.textbox.setText("等待识别中。。。。")
    # 按键按下
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        if event.button() == Qt.RightButton:
            self.sezelog = True
            self.stars = event.globalPos()
    # 按键移动
    def mouseMoveEvent(self, event):
        if self.is_dragging:
            new_position = event.globalPos() - self.drag_position
            self.move(new_position)
            self.zimuXXX=new_position.x()
            self.zimuYYY=new_position.y()
            event.accept()
        elif self.sezelog and event.buttons() == Qt.RightButton:
            newkuang = event.globalPos().x() - self.stars.x()
            newgao = event.globalPos().y() - self.stars.y()
            self.TTT = self.data.get("字幕窗口W") + newkuang
            self.DDD = self.data.get("字幕窗口H") + newgao
            if self.TTT < 100:
                self.TTT = 100
            if self.DDD < 50:
                self.DDD = 50
            self.textbox.resize(self.TTT, self.DDD)
            event.accept()
    # 按键释放
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self.config.update_item("字幕窗口X",self.zimuXXX)
            self.config.update_item("字幕窗口Y",self.zimuYYY)
        elif event.button() == Qt.RightButton:
            self.sezelog = False
            self.config.update_item("字幕窗口W", self.TTT)
            self.config.update_item("字幕窗口H", self.DDD)
    #鼠标右键菜单
    def contextMenuEvent(self, event):
        event.accept()
#连续识别OCR工作线程
class ContinuousThread(QThread):
    def __init__(self,ui_self):
        super(ContinuousThread, self).__init__()
        self.data=ConfigManager()
        self.similarity=float(self.data.get_item("连续识别相似度"))
        self.sleeptime=float(self.data.get_item("连续识别间隔"))
        self.runlog=True
        self.ui_self=ui_self
        self.next=True
        self.last_image=None
    def run(self):
        if self.sleeptime<=0.2:
            times = 0.2
        else:
            times = self.sleeptime
        while self.runlog:
            if self.next:
                image = self.ui_self.capture_screen_area()
                if self.calculate_image_similarity_from_qpixmap(self.last_image,image)<=self.similarity:
                    self.ui_self.ocr_service.task_queue.put(image)
                    self.last_image = image
                    self.next = False
            time.sleep(times)
    def quitthis(self):
        self.runlog=False
    #将QPixmap转换为PIL Image对象
    def qt_pixmap_to_pil_image(self,qt_pixmap):
        qi = QImage(qt_pixmap.toImage())
        if qi.format() == QImage.Format_ARGB32:
            qi = qi.convertToFormat(QImage.Format_RGB32)
        size_tuple = (qi.width(), qi.height())
        mode = "RGB" if qi.depth() >= 24 else "L"
        pil_image = Image.frombytes(mode, size_tuple, qi.bits().asstring(qi.byteCount()))
        return pil_image
    #计算两个QPixmap对象的结构相似度指数(SSIM)，并将其归一化到0到1的范围。
    def calculate_image_similarity_from_qpixmap(self,pixmap1, pixmap2,win_size=None):
        if pixmap1==None or pixmap2==None:
            return 0.1
        else:
            img1 = self.qt_pixmap_to_pil_image(pixmap1)
            img2 = self.qt_pixmap_to_pil_image(pixmap2)
            # 确保两个图像尺寸相同
            img1, img2 = self.resize_images_to_same_dim(img1, img2)
            img1_array = np.array(img1)
            img2_array = np.array(img2)
            multichannel = img1_array.ndim > 2
            ssim_value = ssim(img1_array, img2_array, win_size=win_size, data_range=img2_array.max() - img2_array.min(),multichannel=multichannel)
            normalized_ssim = (ssim_value + 1) / 2
            return round(normalized_ssim, 2)
    #调整两个图像的尺寸至相同，保持长宽比并填充或裁剪以匹配尺寸
    def resize_images_to_same_dim(self,image1, image2):
        size1 = image1.size
        size2 = image2.size
        if size1 == size2:
            return image1, image2
        # 确定目标尺寸为两者中的较大者
        target_size = max(size1, size2)
        # 使用Image.Resampling.BILINEAR或Image.Resampling.BICUBIC进行高质量的图像缩放
        resized_image1 = image1.resize(target_size, resample=Image.Resampling.BICUBIC)
        resized_image2 = image2.resize(target_size, resample=Image.Resampling.BICUBIC)
        return resized_image1, resized_image2
#主窗口
class OCRMainWindow(QMainWindow):
    # 快捷键信号
    global zuobiao
    zuobiao=ContinuousRecognize_xy()
    ocrhotkey = pyqtSignal()
    prtschotkey = pyqtSignal()
    subtitleshotkey = pyqtSignal()
    #初始化主界面，以及信号
    def __init__(self):
        super().__init__()
        # 设置窗口标题
        self.setWindowTitle("OCR翻译工具")
        self.setTrayIcon()
        self.setWindowIcon(QIcon("./colorful_logo.ico"))
        # 主布局为水平分割布局黄金分割
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.splitter = QSplitter(Qt.Horizontal)
        self.central_widget.setLayout(QHBoxLayout())
        self.central_widget.layout().addWidget(self.splitter)
        # 设置窗口的最小尺寸
        self.setMinimumSize(600, 500)
        # 左边部分，垂直布局，包含两个标签和对应的文本框
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        # 添加描述性标签
        self.recognition_label = QLabel("【识别结果 或 需翻译文本】：")
        self.translation_label = QLabel("【翻译结果】：")
        self.recognition_text = QTextEdit()
        self.translation_text = QTextEdit()
        # 将标签和文本框添加到布局中
        left_layout.addWidget(self.recognition_label)
        left_layout.addWidget(self.recognition_text)
        left_layout.addWidget(self.translation_label)
        left_layout.addWidget(self.translation_text)
        left_widget.setLayout(left_layout)
        self.splitter.addWidget(left_widget)
        self.splitter.setStretchFactor(0, 2)#左边占比较大
        # 右边部分，垂直布局，包含多个按钮和开关
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        self.start_button = QPushButton("OCR识别")
        self.continuous_recognize_button = QPushButton("OCR连续识别")
        self.movie_subtitles_button=QPushButton("电影字幕连续识别")
        self.translate_button = QPushButton("中英文本翻译")
        self.screenshot_button = QPushButton("屏幕截图")
        self.settings_button = QPushButton("设置")
        self.ocr_engine_switch = QCheckBox("OCR引擎")
        self.translation_engine_switch = QCheckBox("翻译引擎")
        self.show_translation_switch = QCheckBox("识别后翻译")
        self.show_subtitles_switch=QCheckBox("显示字幕")
        for button in [
            self.start_button,self.continuous_recognize_button,self.movie_subtitles_button,self.translate_button,self.screenshot_button,
            self.settings_button,self.ocr_engine_switch,self.translation_engine_switch,self.show_translation_switch,
            self.show_subtitles_switch
        ]:
            right_layout.addWidget(button)
        right_widget.setLayout(right_layout)
        self.splitter.addWidget(right_widget)
        self.splitter.setStretchFactor(1, 1)#右边占比较小的部分
        #加载config及项目的设置
        self.init_seting()
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        #绑定按键
        self.start_button.clicked.connect(self.on_start_ocr_clicked)
        self.continuous_recognize_button.clicked.connect(self.on_continuous_recognize_clicked)
        self.movie_subtitles_button.clicked.connect(self.movie_continuous_recognize_clicked)
        self.translate_button.clicked.connect(self.on_translate_clicked)
        self.screenshot_button.clicked.connect(self.on_screenshot_clicked)
        self.settings_button.clicked.connect(self.settings_dialog)
        self.ocr_engine_switch.stateChanged.connect(self.on_ocr_engine_state_changed)
        self.translation_engine_switch.stateChanged.connect(self.on_translation_engine_state_changed)
        self.show_translation_switch.stateChanged.connect(self.on_show_translation_state_changed)
        self.show_subtitles_switch.stateChanged.connect(self.on_show_subtitles_state_changed)
        self.ocr_cachetext=""
        self.continuous_recognize_button_isrun=False
        self.move_mode=False
    # 快捷键槽：转化热键信号为QT信号，如果直接触发则会没有对象self，无法运行进程类函数
    def sendkey1(self):
        self.ocrhotkey.emit()
    # 快捷键槽
    def sendkey2(self):
        self.prtschotkey.emit()
    # 快捷键槽
    def sendkey3(self):
        self.subtitleshotkey.emit()
    #加载配置
    def init_seting(self):
        self.config=ConfigManager()
        self.copy_to_clipboard=bool(self.config.get_item("识别结果复制到剪贴板"))  # 复制到剪切板
        self.subtitles_font_size=int(self.config.get_item("字幕字体大小"))  # 字幕大小
        self.continuous_recognition_interval=float(self.config.get_item("连续识别间隔"))  # 连续识别间隔
        self.screens_hot_key1=self.config.get_item("截图快捷键1")  # 截图快捷键1
        self.screens_hot_key2=self.config.get_item("截图快捷键2")  # 截图快捷键2
        self.ocr_hot_key1=self.config.get_item("识别快捷键1")  # ocr快捷键1
        self.ocr_hot_key2=self.config.get_item("识别快捷键2")  # ocr快捷键2
        self.subtitles_hot_key1=self.config.get_item("字幕快捷键1")  # 字幕快捷键1
        self.subtitles_hot_key2=self.config.get_item("字幕快捷键2")  # 字幕快捷键2
        self.ocr_model_path=self.config.get_item("OCR路径")  # ocr路径
        self.ocr_isrun_log=bool(self.config.get_item("OCR引擎"))  # OCR引擎开关
        self.translation_isrun_log=bool(self.config.get_item("翻译引擎"))  # 翻译引擎开关
        self.subtitles_isshow_log=bool(self.config.get_item("显示字幕"))  # 显示字幕开关
        self.translation_after_ocr=bool(self.config.get_item("识别后翻译"))  # 识别后翻译
        self.subtitlesX=int(self.config.get_item("字幕窗口X"))  # 字幕窗口X
        self.subtitlesY=int(self.config.get_item("字幕窗口Y"))  # 字幕窗口Y
        self.subtitlesW=int(self.config.get_item("字幕窗口W"))  # 字幕窗口W
        self.subtitlesH=int(self.config.get_item("字幕窗口H"))  # 字幕窗口H
        self.window_text_size=int(self.config.get_item("窗口字体大小"))
        self.recognition_text.setFontPointSize(self.window_text_size)
        self.translation_text.setFontPointSize(self.window_text_size)
        if self.ocr_isrun_log:
            self.ocr_engine_switch.setChecked(True)
            self.ocr_service = OcrThread()
            self.ocr_service.start()
            self.ocr_service.ocr_result_signal.connect(self.ocr_service_callback)
        else:
            self.ocr_engine_switch.setChecked(False)
        if self.translation_isrun_log:
            self.translation_engine_switch.setChecked(True)
            self.tran_service = TranThread()
            self.tran_service.start()
            self.tran_service.tran_result_signal.connect(self.tran_service_callback)
            self.show_translation_switch.setEnabled(True)
        else:
            self.show_translation_switch.setEnabled(False)
        if self.translation_after_ocr:
            self.show_translation_switch.setChecked(True)
        else:
            self.show_translation_switch.setChecked(False)
        if self.subtitles_isshow_log:
            self.show_subtitles_switch.setChecked(True)
            self.on_show_subtitles_state_changed(2)
        else:
            self.show_subtitles_switch.setChecked(False)
        self.setstatus()#状态栏
        # # 截图快捷键
        self.prtschotkey.connect(self.on_screenshot_clicked)
        self.prtsc_start = SystemHotkey()
        self.prtsc_start.register((self.screens_hot_key1, self.screens_hot_key2), callback=lambda x: self.sendkey2())
        # OCR快捷键
        self.ocrhotkey.connect(self.on_start_ocr_clicked)
        self.ocr_start = SystemHotkey()
        self.ocr_start.register((self.ocr_hot_key1, self.ocr_hot_key2), callback=lambda x: self.sendkey1())
        # 字幕快捷键
        self.subtitleshotkey.connect(self.on_show_subtitles_windows)
        self.sub_start = SystemHotkey()
        self.sub_start.register((self.subtitles_hot_key1, self.subtitles_hot_key2), callback=lambda x: self.sendkey3())
    #刷新设置
    def refresh_setting(self):
        self.config = ConfigManager()
        self.copy_to_clipboard = bool(self.config.get_item("识别结果复制到剪贴板"))  # 复制到剪切板
        self.subtitles_font_size = int(self.config.get_item("字幕字体大小"))  # 字幕大小
        self.continuous_recognition_interval = float(self.config.get_item("连续识别间隔"))  # 连续识别间隔
        self.screens_hot_key1 = self.config.get_item("截图快捷键1")  # 截图快捷键1
        self.screens_hot_key2 = self.config.get_item("截图快捷键2")  # 截图快捷键2
        self.ocr_hot_key1 = self.config.get_item("识别快捷键1")  # ocr快捷键1
        self.ocr_hot_key2 = self.config.get_item("识别快捷键2")  # ocr快捷键2
        self.subtitles_hot_key1 = self.config.get_item("字幕快捷键1")  # 字幕快捷键1
        self.subtitles_hot_key2 = self.config.get_item("字幕快捷键2")  # 字幕快捷键2
        self.ocr_model_path = self.config.get_item("OCR路径")  # ocr路径
        self.ocr_isrun_log = bool(self.config.get_item("OCR引擎"))  # OCR引擎开关
        self.translation_isrun_log = bool(self.config.get_item("翻译引擎"))  # 翻译引擎开关
        self.subtitles_isshow_log = bool(self.config.get_item("显示字幕"))  # 显示字幕开关
        self.translation_after_ocr = bool(self.config.get_item("识别后翻译"))  # 识别后翻译
        self.subtitlesX = int(self.config.get_item("字幕窗口X"))  # 字幕窗口X
        self.subtitlesY = int(self.config.get_item("字幕窗口Y"))  # 字幕窗口Y
        self.subtitlesW = int(self.config.get_item("字幕窗口W"))  # 字幕窗口W
        self.subtitlesH = int(self.config.get_item("字幕窗口H"))  # 字幕窗口H
        self.window_text_size = int(self.config.get_item("窗口字体大小"))
        self.recognition_text.setFontPointSize(self.window_text_size)
        self.translation_text.setFontPointSize(self.window_text_size)
        self.setstatus()  # 状态栏
    #实现OCR识别截图
    def on_start_ocr_clicked(self):
        if self.ocr_isrun_log:
            self.capture_widget = Screenshot(mode=1)
            self.capture_widget.show()
            self.capture_widget.screenshotTaken.connect(self.on_start_ocr_clicked_screenshot)
        else:
            QMessageBox.warning(self, '警告', '请先启动OCR引擎！', QMessageBox.Ok)
    #截图回调OCR单次模式
    def on_start_ocr_clicked_screenshot(self,pixmap):
        self.ocr_service.task_queue.put(pixmap)
    # 连续OCR识别
    def on_continuous_recognize_clicked(self):
        if  self.continuous_recognize_button_isrun==False:
            if self.ocr_isrun_log:
                self.continuous_recognize_button.setText("退出连续识别")
                self.continuous_recognize_button_isrun =True
                self.capture_widget = Screenshot(mode=3)
                self.capture_widget.show()
                self.capture_widget.continuousRecognizelog.connect(self.on_continuous_recognize_clicked_callback)
            else:
                QMessageBox.warning(self, '警告', '请先启动OCR引擎！', QMessageBox.Ok)
        else:
            try:
                self.continuous_recognize_button.setText("OCR连续识别")
                self.continuous_recognize_button_isrun =False
                self.continuous_recognize.close()
                self.continuous_recognize_thread.quitthis()
                self.continuous_recognize_thread.quit()
                self.continuous_recognize_thread.wait()
            except:
                pass
    #连续识别截图后的回调，会启动工作线程
    def on_continuous_recognize_clicked_callback(self):
        if self.continuous_recognize_button_isrun or self.move_mode:
            self.continuous_recognize = ContinuousRecognizeWindows(zuobiao.prtX, zuobiao.prtY, zuobiao.prtW,zuobiao.prtH)
            self.continuous_recognize.show()
            self.continuous_recognize_thread = ContinuousThread(self)
            self.continuous_recognize_thread.start()
    #电影字幕功能
    def movie_continuous_recognize_clicked(self):
        if  self.move_mode==False:
            if self.translation_isrun_log and self.ocr_isrun_log:
                self.move_mode=True
                if not self.subtitles_isshow_log:
                    self.show_subtitles_switch.setChecked(True)
                if not self.translation_after_ocr:
                    self.show_translation_switch.setChecked(True)
                self.movie_subtitles_button.setText("退出电影连续识别")
                self.capture_widget = Screenshot(mode=3)
                self.capture_widget.show()
                self.capture_widget.continuousRecognizelog.connect(self.on_continuous_recognize_clicked_callback)
            else:
                QMessageBox.warning(self, '警告', '请先启动OCR和翻译引擎！', QMessageBox.Ok)
        else:
            try:
                self.move_mode = False
                self.movie_subtitles_button.setText("电影字幕连续识别")
                self.continuous_recognize.close()
                self.continuous_recognize_thread.quitthis()
                self.continuous_recognize_thread.quit()
                self.continuous_recognize_thread.wait()
            except:
                pass
    # 实现文字翻译的逻辑
    def on_translate_clicked(self):
        if self.translation_isrun_log:
            self.ocr_cachetext = ""
            self.tran_service.task_queue.put(self.recognition_text.toPlainText())
        else:
            QMessageBox.warning(self, '警告', '请先启动翻译引擎！', QMessageBox.Ok)
    # 实现屏幕截图的逻辑
    def on_screenshot_clicked(self):
        self.capture_widget = Screenshot(mode=2)
        self.capture_widget.show()
    # OCR引擎开关状态变化
    def on_ocr_engine_state_changed(self, state):
        if state == 2:
            self.ocr_service = OcrThread()
            self.ocr_service.start()
            self.ocr_service.ocr_result_signal.connect(self.ocr_service_callback)
            self.config.update_item("OCR引擎", True)
            self.ocr_isrun_log = True
        elif state == 0:
            self.ocr_service.ocr_run_signal=False
            self.ocr_service.exit()
            self.config.update_item("OCR引擎", False)
            self.ocr_isrun_log = False
        else:
            pass
        self.setstatus()
    # 翻译引擎开关状态变化
    def on_translation_engine_state_changed(self, state):
        if state == 2:
            self.tran_service = TranThread()
            self.tran_service.start()
            self.tran_service.tran_result_signal.connect(self.tran_service_callback)
            self.config.update_item("翻译引擎", True)
            self.translation_isrun_log = True
            self.show_translation_switch.setEnabled(True)
        elif state == 0:
            self.tran_service.tran_run_signal=False
            self.tran_service.exit()
            self.config.update_item("翻译引擎", False)
            self.translation_isrun_log = False
            self.show_translation_switch.setChecked(False)
            self.show_translation_switch.setEnabled(False)
            self.translation_after_ocr = False
        else:
            pass
        self.setstatus()
    # 识别后翻译开关状态变化
    def on_show_translation_state_changed(self, state):
        if state == 2:
            self.config.update_item("识别后翻译", True)
            self.translation_after_ocr = True
        elif state == 0:
            self.config.update_item("识别后翻译", False)
            self.translation_after_ocr = False
        else:
            pass
    # 显示字幕开关状态变化
    def on_show_subtitles_state_changed(self, state):
        if state==2:
            self.subtitleswindows=Subtitleswindows()
            self.subtitleswindows.show()
            self.config.update_item("显示字幕",True)
            self.subtitles_isshow_log=True
        elif state==0:
            self.subtitleswindows.close()
            self.config.update_item("显示字幕", False)
            self.subtitles_isshow_log=False
        else:
            pass
    #快捷键显示隐藏字幕窗口
    def on_show_subtitles_windows(self):
        if self.subtitles_isshow_log:
            self.show_subtitles_switch.setChecked(False)
        else:
            self.show_subtitles_switch.setChecked(True)
    #OCR服务的回调函数
    def ocr_service_callback(self,res):
        zuobiao.overlog=True
        if self.ocr_service.ocr_run_signal:
            if self.copy_to_clipboard:
                self.clipboard = QApplication.clipboard()# 获取应用程序的剪贴板
                self.clipboard.setText(res)# 将字符串设置到剪贴板上
            if self.translation_after_ocr:
                self.ocr_cachetext=res
                self.tran_service.task_queue.put(res)
            else:
                try:
                    self.continuous_recognize_thread.next=True
                except:
                    pass
                if self.subtitles_isshow_log:
                    self.recognition_text.setText(res)
                    self.subtitleswindows.textbox.setText(res)
                else:
                    self.recognition_text.setText(res)
                    self.show()
                    self.activateWindow()
    #翻译服务回调函数
    def tran_service_callback(self,res):
        if self.tran_service.tran_run_signal:
            try:
                self.continuous_recognize_thread.next = True
            except:
                pass
            if self.subtitles_isshow_log:
                if self.move_mode:
                    self.subtitleswindows.textbox.setText(res)
                else:
                    if self.ocr_cachetext!=None and self.ocr_cachetext!="":
                        self.recognition_text.setText(self.ocr_cachetext)
                    self.translation_text.setText(res)
                    self.subtitleswindows.textbox.setText("【识别】\n"+self.ocr_cachetext+"\n"+"【翻译】\n"+res)
                    self.ocr_cachetext = ""
            else:
                if self.ocr_cachetext!=None and self.ocr_cachetext!="":
                    self.recognition_text.setText(self.ocr_cachetext)
                self.translation_text.setText(res)
                self.show()
                self.activateWindow()
    #获取连续识别的截图pixmap对象
    def capture_screen_area(self):
        screen = QApplication.primaryScreen()
        pixmap = screen.grabWindow(QApplication.desktop().winId(), int(zuobiao.prtX),int(zuobiao.prtY),int(zuobiao.prtW),int(zuobiao.prtH))
        return pixmap
    #状态栏
    def setstatus(self):
        if self.ocr_isrun_log==True and self.translation_isrun_log==True:
            status="【OCR引擎已启动】【翻译引擎已启动】"
        elif self.ocr_isrun_log==True and self.translation_isrun_log==False:
            status = "【OCR引擎已启动】【翻译引擎未启动】"
        elif self.ocr_isrun_log == False and self.translation_isrun_log == True:
            status = "【CR引擎未启动】【翻译引擎已启动】"
        else:
            status="【OCR引擎未启动】【翻译引擎未启动】"
        self.statusBar().showMessage(status)
    # 判断字符串是否匹配RE
    def assert_matches_regex(self,string, pattern):
        match = re.match(pattern, string)
        if match == None:
            return False
        else:
            return True
    #设置界面
    def settings_dialog(self):
        class SettingsDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                set_config = ConfigManager()
                self.setWindowTitle("设置")
                layout = QVBoxLayout()
                form_layout = QFormLayout()
                # 隐藏右上角的问号按钮
                self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                self.label=QLabel()
                self.label.setText("注：快捷键设置后需重启才能生效!!!")
                self.label.setStyleSheet("color: red;")
                form_layout.addRow(self.label)
                # 到剪贴板（开关）
                self.clipboard_switch = QComboBox(self)
                self.clipboard_switch.addItem("开启")
                self.clipboard_switch.addItem("关闭")
                form_layout.addRow("识别结果复制到剪贴板", self.clipboard_switch)
                if set_config.get_item("识别结果复制到剪贴板")==True:
                    self.clipboard_switch.setCurrentText("开启")
                else:
                    self.clipboard_switch.setCurrentText("关闭")

                # 字幕字体字号（选择框）
                self.font_size_spinbox = QSpinBox()
                self.font_size_spinbox.setRange(5, 40)
                form_layout.addRow("字幕字体大小", self.font_size_spinbox)
                self.font_size_spinbox.setValue(set_config.get_item("字幕字体大小"))

                #字幕描边颜色
                self.font_stroke_color = QLineEdit()
                form_layout.addRow("字幕描边颜色(十六进制大写)", self.font_stroke_color)
                self.font_stroke_color.setText(set_config.get_item("字幕描边颜色"))

                #字幕描边宽度
                self.subtitle_stroke_width = QLineEdit()
                form_layout.addRow("字幕描边宽度 (浮点数)", self.subtitle_stroke_width)
                self.subtitle_stroke_width.setText(str(set_config.get_item("字幕描边宽度")))

                #字幕透明度
                self.subtitl_transparency = QLineEdit()
                form_layout.addRow("字幕透明度 (浮点数)", self.subtitl_transparency)
                self.subtitl_transparency.setText(str(set_config.get_item("字幕透明度")))

                # 连续识别间隔（输入框）
                self.recognition_interval = QLineEdit()
                form_layout.addRow("连续识别间隔 (浮点数)", self.recognition_interval)
                self.recognition_interval.setText(str(set_config.get_item("连续识别间隔")))

                # 连续识别相似度
                self.Image_similarity = QLineEdit()
                form_layout.addRow("连续识别相似度 (浮点数)", self.Image_similarity)
                self.Image_similarity.setText(str(set_config.get_item("连续识别相似度")))

                # 截图快捷键（两个输入框）
                self.screenshot_key1 = QComboBox(self)
                self.screenshot_key1.addItem("ctrl")
                self.screenshot_key1.addItem("alt")
                self.screenshot_key2 = QLineEdit()
                form_layout.addRow("截图快捷键1", self.screenshot_key1)
                form_layout.addRow("截图快捷键2(数字/字母)", self.screenshot_key2)
                if set_config.get_item("截图快捷键1")=="ctrl":
                    self.screenshot_key1.setCurrentText("ctrl")
                else:
                    self.screenshot_key1.setCurrentText("alt")
                self.screenshot_key2.setText(set_config.get_item("截图快捷键2"))

                # 识别快捷键（两个输入框）
                self.recognition_key1 = QComboBox(self)
                self.recognition_key1.addItem("ctrl")
                self.recognition_key1.addItem("alt")
                self.recognition_key2 = QLineEdit()
                form_layout.addRow("识别快捷键1", self.recognition_key1)
                form_layout.addRow("识别快捷键2(数字/字母)", self.recognition_key2)
                if set_config.get_item("识别快捷键1")=="ctrl":
                    self.recognition_key1.setCurrentText("ctrl")
                else:
                    self.recognition_key1.setCurrentText("alt")
                self.recognition_key2.setText(set_config.get_item("识别快捷键2"))

                #字幕快捷键（两个输入框）
                self.subtitle_key1 = QComboBox(self)
                self.subtitle_key1.addItem("ctrl")
                self.subtitle_key1.addItem("alt")
                self.subtitle_key2 = QLineEdit()
                form_layout.addRow("字幕快捷键1", self.subtitle_key1)
                form_layout.addRow("字幕快捷键2(数字/字母)", self.subtitle_key2)
                if set_config.get_item("字幕快捷键1")=="ctrl":
                    self.subtitle_key1.setCurrentText("ctrl")
                else:
                    self.subtitle_key1.setCurrentText("alt")
                self.subtitle_key2.setText(set_config.get_item("字幕快捷键2"))

                #窗口字体大小
                self.window_font_size = QSpinBox()
                self.window_font_size.setRange(5, 40)
                form_layout.addRow("窗口字体大小", self.window_font_size)
                self.window_font_size.setValue(set_config.get_item("窗口字体大小"))

                # 9. OCR路径（较长输入框）
                self.ocr_path = QLineEdit()
                form_layout.addRow("OCR路径(path)", self.ocr_path)
                self.ocr_path.setText(set_config.get_item("OCR路径"))

                # 保存设置按钮
                button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
                button_box.accepted.connect(self.accept)
                button_box.rejected.connect(self.reject)
                layout.addLayout(form_layout)
                layout.addWidget(button_box)
                self.setLayout(layout)

        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            set_config=ConfigManager()
            if dialog.clipboard_switch.currentText()=="开启":
                set_config.update_item("识别结果复制到剪贴板",True)
            else:
                set_config.update_item("识别结果复制到剪贴板", False)
            try:
                assert isinstance(float(dialog.recognition_interval.text()),float)
                assert self.assert_matches_regex(dialog.screenshot_key2.text(),"^[a-zA-Z0-9]$")
                assert self.assert_matches_regex(dialog.recognition_key2.text(), "^[a-zA-Z0-9]$")
                assert self.assert_matches_regex(dialog.subtitle_key2.text(), "^[a-zA-Z0-9]$")
                assert self.assert_matches_regex(dialog.font_stroke_color.text(),"^#[A-F0-9]{6}$")
                set_config.update_item("字幕字体大小",dialog.font_size_spinbox.value())
                set_config.update_item("字幕描边颜色", dialog.font_stroke_color.text())
                set_config.update_item("字幕描边宽度", float(dialog.subtitle_stroke_width.text()))
                set_config.update_item("字幕透明度", float(dialog.subtitl_transparency.text()))
                set_config.update_item("连续识别间隔", float(dialog.recognition_interval.text()))
                set_config.update_item("连续识别相似度", float(dialog.Image_similarity.text()))
                set_config.update_item("截图快捷键1",dialog.screenshot_key1.currentText())
                set_config.update_item("截图快捷键2", dialog.screenshot_key2.text())
                set_config.update_item("识别快捷键1", dialog.recognition_key1.currentText())
                set_config.update_item("识别快捷键2", dialog.recognition_key2.text())
                set_config.update_item("字幕快捷键1", dialog.subtitle_key1.currentText())
                set_config.update_item("字幕快捷键2", dialog.subtitle_key2.text())
                set_config.update_item("OCR路径",dialog.ocr_path.text())
                set_config.update_item("窗口字体大小",dialog.window_font_size.value())
            except:
                checkFlag = QtWidgets.QMessageBox.information(self, "警告", "设置输入值类型错误，请重新设置！")
            self.refresh_setting()
    # 托盘栏
    def setTrayIcon(self):
        # 初始化菜单单项
        self.openwin_menu = QAction("显示主窗口")
        self.openset_menu = QAction("设置")
        self.quit_menu = QAction("退出")
        # 菜单单项连接方法
        self.openwin_menu.triggered.connect(self.show_wind)
        self.openset_menu.triggered.connect(self.settings_dialog)
        self.quit_menu.triggered.connect(self.app_quit)
        # 初始化菜单列表
        self.trayIconMenu = QMenu()
        self.trayIconMenu.addAction(self.openwin_menu)
        self.trayIconMenu.addAction(self.openset_menu)
        self.trayIconMenu.addSeparator()
        self.trayIconMenu.addAction(self.quit_menu)
        # 构建菜单UI
        self.trayIcon = QtWidgets.QSystemTrayIcon()
        self.trayIcon.setContextMenu(self.trayIconMenu)
        self.trayIcon.setIcon(QIcon("./white_logo.ico"))
        self.trayIcon.setToolTip("双击打开窗口")
        # 左键双击打开主界面
        self.trayIcon.activated[QtWidgets.QSystemTrayIcon.ActivationReason].connect(self.openMainWindow)
        # 允许托盘菜单显示
        self.trayIcon.show()
    # 托盘图标双击显示主窗口
    def openMainWindow(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.DoubleClick:
            self.show_wind()
    # 显示主窗口
    def show_wind(self):
        self.showNormal()
        self.activateWindow()
    # 图标退出&二次确认
    def app_quit(self):
        checkFlag = QtWidgets.QMessageBox.information(self, "提示", "是否确认退出？",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if checkFlag == QtWidgets.QMessageBox.Yes:
            if self.ocr_isrun_log:
                self.ocr_service.quitocr()
                while self.ocr_service.isRunning():
                    time.sleep(0.1)
            elif self.translation_isrun_log:
                self.tran_service.tran_run_signal = False
                while self.tran_service.isRunning():
                    time.sleep(0.1)
                self.tran_service.exit()
            else:
                pass
            QtWidgets.qApp.quit()
    @staticmethod #桌面路径
    def getdesktop():
        key = win32api.RegOpenKey(win32con.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders', 0,win32con.KEY_READ)
        return win32api.RegQueryValueEx(key, 'Desktop')[0]
#入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = OCRMainWindow()
    sys.exit(app.exec_())

