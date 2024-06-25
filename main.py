import os
import shutil
from PyQt5.QtWidgets import QApplication, QMessageBox, QFileDialog
from PyQt5 import uic
from PyQt5.QtCore import QFile, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

# Yolov7 import
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

'''
    Designed by LJY
    Referenced by 
'''

ui_path = "ui/v7.ui"
c_ui_path = "ui/notice.ui"
ui_icon = "ui/img.png"
cache = 'cache/'


# 主ui定义
class Stats:
    def __init__(self):
        self.ui = uic.loadUi(ui_path)
        self.child_ui = ChildWindow()
        self.parser_init()

        # 信号与槽连接关系
        self.ui.action_model.triggered.connect(self.load_model)
        self.ui.toolButton_choosefile.clicked.connect(self.choosefile)
        self.ui.toolButton_save.clicked.connect(self.savefile)
        self.ui.comboBox_readfile.currentIndexChanged.connect(self.showsrcimg)
        self.ui.comboBox_choose_item.addItems(['图片检测', '视频检测', '摄像头检测'])
        self.ui.comboBox_choose_item.currentIndexChanged.connect(self.choose_item)
        self.ui.toolButton_choosefile_video.clicked.connect(self.choose_video_file)

        self.ui.pushButton_del_parameter.clicked.connect(self.clear_parameter)
        self.ui.pushButton_del_cache.clicked.connect(self.del_cache)
        self.ui.pushButton_save.clicked.connect(self.save)
        self.ui.pushButton_saveall.clicked.connect(self.save_all)
        self.ui.pushButton_test.clicked.connect(self.open_childwindow)
        self.ui.pushButton_video_pause.clicked.connect(self.video_pause)
        self.ui.pushButton_video_finish.clicked.connect(self.finish_detect)
        self.ui.pushButton_video_begin.clicked.connect(self.video_begin)
        self.ui.pushButton_cam_select.clicked.connect(self.search_cam_device)
        self.ui.pushButton_cam_begin.clicked.connect(self.camera_detect)
        self.ui.pushButton_cam_stop.clicked.connect(self.camera_disabled)

        # 功能选择初始化
        self.mode = 0
        self.ui.stackedWidget_item.setCurrentIndex(self.mode)

        # 初始化权重名
        self.openfile_name_model = ('', '')

        # 初始化参数框及参数
        self.labels = []
        self.show_info_init()  # 在此初始化 self.parameters
        self.show_info_()

        # 目录变量
        self.filepath = ''  # 读取图片父目录
        self.savepath = ''  # 保存图片父目录
        self.cachepath = ''  # 在当前目录下创建缓存文件夹
        self.currimg = ''  # 在缓存目录下的文件目录
        self.filepathlist = []  # 源图片目录列表

        # 计数变量
        self.count = 0
        self.act_count = 0
        self.ui.save_progressbar.reset()

        # 视频变量
        self.cap = cv2.VideoCapture()
        self.video_file = ('', '')
        self.out = None
        self.video_timer = QTimer()
        self.num_stop = 1
        self.ui.textBrowser_video.clear()

        # 摄像头变量
        self.select_cam_num = -1
        self.cam_max = 10000
        self.cam_num = 0
        self.ui.comboBox_cam_num.addItem(str(self.select_cam_num))

        # 缓存目录初始化
        self.mkdir()

    def parser_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='D:/DL_files/aircraft_result/ac5_200_16/weights/best.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='D:/DL_files/aircraft_skin_6/val/images',
                            help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf_thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou_thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view_img', action='store_true', help='display results')
        parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='Yolov7', help='save results to project/name')
        parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no_trace', action='store_true', help='don`t trace model')
        parser.add_argument('--line_thickness', type=int, default=5, help='set the line thickness')
        self.opt = parser.parse_args()

    def load_model(self):
        view_img, save_txt, imgsz = self.opt.view_img, self.opt.save_txt, self.opt.img_size

        self.openfile_name_model = QFileDialog.getOpenFileName(self.ui, "选择权重文件路径")
        # print(self.openfile_name_model)
        if self.openfile_name_model[0] != '' and self.openfile_name_model[0].endswith('.pt'):
            weights = self.openfile_name_model[0]
        else:
            self.openfile_name_model = ('', '')
            QMessageBox.critical(self.ui, '路径有误', '选取权重文件格式或路径有误!')
            return

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True
        try:
            self.model = attempt_load(weights, map_location=self.device)
            stride = int(self.model.stride.max())
            self.imgsz = check_img_size(imgsz, s=stride)
            if self.half:
                self.model.half()
            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            # 修改状态信息
            self.parameters = '模型状态：权重文件 {} 加载成功\n----------------------------------------------------------' \
                              '----------------'.format(weights)
            self.show_info_()
            # print("model initial done")
            QMessageBox.information(self.ui, "提示", "模型初始化成功")
        except AttributeError:
            QMessageBox.warning(self.ui, "警告", "请加载 Yolo v7 模型! ")
            return

    # 新建文件夹函数
    def mkdir(self):
        rootpath = os.getcwd()
        is_folder = os.path.exists(cache)
        if not is_folder:
            os.mkdir(cache)
        self.cachepath = rootpath + os.path.sep + 'cache'
        # print(self.cachepath)

    # 文件夹选择槽函数
    def choosefile(self):
        self.filepath = ''
        self.currimg = ''
        self.filepathlist = []
        self.count = 0
        self.ui.save_progressbar.reset()
        self.ui.comboBox_readfile.clear()
        self.ui.showimg.clear()
        self.ui.showimg_c.clear()
        self.filepath = QFileDialog.getExistingDirectory(self.ui, "选择图片路径")
        if len(self.filepath) != 0:
            for filename in os.listdir(self.filepath):
                if filename.endswith(".jpg"):
                    self.ui.comboBox_readfile.addItem(filename)
                    self.filepathlist.append(self.filepath + '/' + filename)
                    self.count += 1
        else:
            self.ui.showimg.setText('原图片显示区域')
            self.ui.showimg_c.setText('处理后显示图片')

    # 选择框槽函数
    def choose_item(self):
        self.select_cam_num = -1
        self.video_timer.stop()
        self.clear_parameter()
        self.ui.label_video.setText('视频显示区域')
        self.ui.label_cam.setText('摄像头检测')
        try:
            self.cap.release()
            self.out.release()
        except AttributeError:
            pass

        item_choose = self.ui.comboBox_choose_item.currentText()
        self.mode = 0 if item_choose == '图片检测' else 1 if item_choose == '视频检测' else 2 if item_choose == '摄像头检测' else 3

        self.ui.stackedWidget_item.setCurrentIndex(self.mode)

        # todo camera process
        if self.filepath != '' and self.mode == 0:
            self.showsrcimg()
        elif self.mode == 1:
            pass
        elif self.mode == 2:
            pass

    # 图片检测函数
    def detect_pic(self, path):
        self.labels = []
        if len(path) != 0:
            img = cv2.imread(path)
            showimg = img
            with torch.no_grad():
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # BGR => RGB
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()
                img /= 255.0  # 0-255 => 0-1 归一化
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                pred = self.model(img, augment=self.opt.augment)[0]
                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # print(pred)
                # Process detections
                for i, det in enumerate(pred):
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            self.labels.append(plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                            line_thickness=self.opt.line_thickness, qt=True))

            # 在参数框中显示检测到的标签
            # print(self.labels)
            self.show_info_()
            fin_path = self.cachepath + os.path.sep + os.path.basename(path)
            # print('fp'+fin_path)
            cv2.imwrite(fin_path, showimg)
            return fin_path

    # 显示图片函数
    def showsrcimg(self):
        currimg_path = self.ui.comboBox_readfile.currentText()
        abs_path = self.filepath + '/' + currimg_path
        if len(abs_path) != 0 and (self.openfile_name_model[0]) != '':
            # print('ap'+abs_path)
            pixmap_src = QPixmap(abs_path)
            self.ui.showimg.setPixmap(pixmap_src)
            self.ui.showimg.setScaledContents(True)
            self.currimg = self.detect_pic(abs_path)
            pixmap_c = QPixmap(self.currimg)
            self.ui.showimg_c.setPixmap(pixmap_c)
            self.ui.showimg_c.setScaledContents(True)
        elif len(abs_path) != 0 and len(self.openfile_name_model[0]) == 0:
            pixmap_src = QPixmap(abs_path)
            self.ui.showimg.setPixmap(pixmap_src)
            self.ui.showimg.setScaledContents(True)
            QMessageBox.warning(self.ui, "警告", "未加载模型文件\n请先进行模型配置!")
        else:
            return

    # 初始化参数框
    def show_info_init(self):
        self.ui.show_info.clear()
        self.parameters = '模型状态：请加载权重模型\n--------------------------------------------------------------------------'

    # 参数框更新
    def show_info_(self):
        self.ui.show_info.clear()
        self.ui.show_info.append(self.parameters)
        if len(self.labels) != 0:
            for x in self.labels:
                self.ui.show_info.append(x)

    # 选择保存路径槽函数
    def savefile(self):
        self.ui.textBrowser_save.clear()
        self.savepath = ''
        self.savepath = QFileDialog.getExistingDirectory(self.ui, "选择保存路径")
        self.ui.textBrowser_save.append(self.savepath)

    # 保存图片槽函数
    def save(self):
        self.ui.save_progressbar.reset()
        self.ui.save_progressbar.setRange(0, 1)
        # print(self.currimg)
        if len(self.currimg) == 0:
            QMessageBox.critical(self.ui, '错误', '图片读取失败！')
        else:
            if len(self.savepath) != 0:
                # print(self.savepath)
                shutil.copyfile(self.currimg, self.savepath + '/' + os.path.basename(self.currimg))
                self.ui.save_progressbar.setValue(1)
                QMessageBox.information(self.ui, '保存成功', '已将图片保存至' + self.savepath + '中')
            else:
                QMessageBox.critical(self.ui, '错误', '保存路径为空！')

    # 保存全部槽函数
    def save_all(self):
        self.ui.save_progressbar.reset()
        self.act_count = 0
        if self.count != 0:
            self.ui.save_progressbar.setRange(0, self.count)
        # print(self.filepathlist)
        if len(self.currimg) == 0:
            QMessageBox.critical(self.ui, '错误', '图片读取失败！')
        else:
            if len(self.savepath) != 0:
                for i in range(self.count):
                    cpf = self.detect_pic(self.filepathlist[i])
                    shutil.copyfile(cpf, self.savepath + '/' + os.path.basename(cpf))
                    self.act_count += 1
                    self.ui.save_progressbar.setValue(self.act_count)
                QMessageBox.information(self.ui, '保存成功', '成功保存了' + str(self.act_count) + '张图片')
            else:
                QMessageBox.critical(self.ui, '错误', '保存路径为空！')

    # 选择视频槽函数
    def choose_video_file(self):
        self.video_file = QFileDialog.getOpenFileName(self.ui, '选择视频文件路径', '')
        if self.video_file[0] != '':
            fin_path = self.cachepath + os.path.sep + os.path.basename(self.video_file[0])
            self.ui.textBrowser_video.setText(self.video_file[0])
            flag = self.cap.open(self.video_file[0])
            if flag:
                frame_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                self.out = cv2.VideoWriter(fin_path, fourcc, frame_num, (frame_width, frame_height))
                # todo change the frame rate
                self.video_timer.start(60)
            else:
                self.video_file = ('', '')
                self.ui.textBrowser_video.clear()
                QMessageBox.critical(self.ui, '错误', '视频文件打开失败!')
                return
        else:
            self.video_file = ('', '')
            return

    def video_begin(self):
        if self.video_file[0] != '':
            self.video_timer.timeout.connect(self.show_video_frame)
        else:
            self.video_warning()
            return

    def video_warning(self):
        QMessageBox.warning(self.ui, '警告', '请选择视频文件')

    # 显示视频帧
    def show_video_frame(self):
        flag, img = self.cap.read()
        self.clear_parameter()
        if img is not None:
            showimg = img
            try:
                with torch.no_grad():
                    img = letterbox(img, new_shape=self.opt.img_size)[0]
                    # Convert
                    # BGR to RGB, to 3x416x416
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(self.device)
                    img = img.half() if self.half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    # Inference
                    pred = self.model(img, augment=self.opt.augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                               agnostic=self.opt.agnostic_nms)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], showimg.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                label = '%s %.2f' % (self.names[int(cls)], conf)
                                self.labels.append(plot_one_box(xyxy, showimg, label=label, color=self.colors[int(cls)],
                                                                line_thickness=self.opt.line_thickness, qt=True))
            except AttributeError:
                self.video_timer.timeout.disconnect(self.show_video_frame)
                QMessageBox.warning(self.ui, '警告', '请先加载权重文件!')
                return

            self.out.write(showimg)
            self.show_info_()
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QImage(self.result.data, self.result.shape[1], self.result.shape[0], QImage.Format_RGB888)
            if self.mode == 1:
                self.ui.label_video.setPixmap(QPixmap.fromImage(showImage))
            elif self.mode == 2:
                self.ui.label_cam.setPixmap(QPixmap.fromImage(showImage))
            else:
                return
        else:
            self.video_timer.stop()
            self.cap.release()
            self.out.release()
            self.ui.label_video.setText('视频显示区域')

    # 暂停/继续 视频
    def video_pause(self):
        if self.video_file[0] != '':
            self.video_timer.blockSignals(False)
            # 暂停检测
            # 若QTimer已经触发，且激活
            if self.video_timer.isActive() == True and self.num_stop % 2 == 1:
                self.ui.pushButton_video_pause.setText('继续')  # 当前状态为暂停状态
                self.num_stop = self.num_stop + 1  # 调整标记信号为偶数
                self.video_timer.blockSignals(True)
            # 继续检测
            else:
                self.num_stop = self.num_stop + 1
                self.ui.pushButton_video_pause.setText('暂停')
        else:
            self.video_warning()
            return

    # 结束视频
    def finish_detect(self):
        if self.video_file[0] != '':
            self.cap.release()  # 释放video_capture资源
            self.out.release()  # 释放video_writer资源
            self.ui.label_video.setText('视频显示区域')
            self.video_timer.timeout.disconnect(self.show_video_frame)
            self.video_file = ('', '')
            self.ui.textBrowser_video.setText(self.video_file[0])
            self.clear_parameter()

            # 结束检测时，查看暂停功能是否复位，将暂停功能恢复至初始状态
            # Note:点击暂停之后，num_stop为偶数状态
            if self.num_stop % 2 == 0:
                self.ui.pushButton_video_pause.setText('暂停')
                self.num_stop = self.num_stop + 1
                self.video_timer.blockSignals(False)
        else:
            self.video_warning()
            return

    # 摄像头设备检测
    def search_cam_device(self):
        self.camera_disabled()
        cnt = 0
        for device in range(self.cam_max):
            stream = cv2.VideoCapture(device, cv2.CAP_DSHOW)
            is_grabbed = stream.grab()
            stream.release()
            if is_grabbed:
                cnt = cnt + 1
            else:
                break
        if cnt == 0:
            self.select_cam_num = -1
            QMessageBox.information(self.ui, '提示', '未检测到摄像头')
        else:
            self.ui.comboBox_cam_num.clear()
            for i in range(cnt):
                self.ui.comboBox_cam_num.addItem(str(i))
            self.cam_num = cnt
            QMessageBox.information(self.ui, '提示', '共检测到 {} 个摄像头'.format(cnt))

    # 摄像头检测
    def camera_detect(self):
        self.select_cam_num = int(self.ui.comboBox_cam_num.currentText())
        if self.select_cam_num != -1:
            fin_path = self.cachepath + os.path.sep + 'camera_current'
            flag = self.cap.open(self.select_cam_num, cv2.CAP_DSHOW)
            if flag:
                frame_num = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                self.out = cv2.VideoWriter(fin_path, fourcc, frame_num, (frame_width, frame_height))
                self.video_timer.start(60)
                self.video_timer.timeout.connect(self.show_video_frame)
            else:
                QMessageBox.critical(self.ui, '错误', '摄像头打开失败')
                return
        else:  # 代码同 camera_disabled()
            self.select_cam_num = -1
            self.video_timer.stop()
            self.clear_parameter()
            self.ui.label_cam.setText('摄像头检测')
            try:
                self.cap.release()
                self.out.release()
            except AttributeError:
                QMessageBox.warning(self.ui, '警告', '请先点击摄像头检测选择设备')

    # 摄像头关闭
    def camera_disabled(self):
        self.select_cam_num = -1
        self.video_timer.stop()
        self.clear_parameter()
        self.ui.label_cam.setText('摄像头检测')
        try:
            self.cap.release()
            self.out.release()
        except AttributeError:
            pass

    # 清除缓存
    def del_cache(self):
        shutil.rmtree(cache)
        self.mkdir()
        self.currimg = ''
        QMessageBox.information(self.ui, '提示信息', '成功清除缓存文件！')

    def open_childwindow(self):
        self.child_ui.ui.exec()

    def clear_parameter(self):
        self.labels = []
        self.show_info_()


# 子ui定义
class ChildWindow:
    def __init__(self):
        self.ui = uic.loadUi(c_ui_path)
        self.ui.pushButton_r.clicked.connect(self.return_main)

    def return_main(self):
        self.ui.close()


class QSSLoader:
    def __init__(self):
        pass

    @staticmethod
    def read_qss_file(qss_file_name):
        with open(qss_file_name, 'r', encoding='UTF-8') as file:
            return file.read()


if __name__ == "__main__":
    # Qt load
    app = QApplication([])
    app.setWindowIcon(QIcon(ui_icon))
    # ---
    # style_file = 'QSS-master/Aqua.qss'
    # style_sheet = QSSLoader.read_qss_file(style_file)
    # ---
    stats = Stats()
    # stats.ui.setStyleSheet(style_sheet)
    stats.ui.show()
    app.exec_()
