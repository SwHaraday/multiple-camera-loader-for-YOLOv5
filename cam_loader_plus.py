# THIS 📷 by SWCC Corporation, GPL-3.0 license
"""
usage :
    dataset = LoadV4TISCams(source, img_size=640, stride=32, auto=True)
"""

import os, sys
import codecs
import time
from threading import Thread
import re
import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore") # Warning will make operation confuse!!!

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    scaleup=True,
    stride=32,
):
    shape = im.shape[:2]  # current shape [height, width]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)

def get_camera_params(source):
    # 引数で渡されたシリアルナンバーのテキストファイルを元に設定値を辞書として返す関数
    source = str(source) # 引数が整数でも耐えるように…
    fn = source + '.txt'
    with codecs.open(fn, 'r', 'utf-8') as f: 
        # 行頭文字が「＃」で無いものを取り込む「キーワード 数値」のリストになる
        params=[x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
    # dict型変数を用意
    p_dict={}
    for x in params: # リストの各要素をスペースで分けてキーワードと数値として書き込む
        # split()は半角/全角スペースが混在しようが、いくつ並んでいようが、関係なくスペースで分けてくれる
        p_dict[x.split()[0]] = x.split()[1] 
    return p_dict

def set_camera_params(p_dict, i, hGrabber, ic, ctypes, tis):
    # 受け取った設定値（辞書型：p_dict）に従って指定されたカメラパラメータを設定する。
    # カメラの露光時間、FPS、ホワイトバランス、ゲインなどを設定する
    if 'Intensity' in p_dict and 'GlobalBrightnessFactor' in p_dict:
        # WDR（ダイナミックレンジを広げて明るくする）をセットしてみる　※撚線機の画質改善のため
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Tone Mapping"), tis.T("Enable"), 1)
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Tone Mapping"), tis.T("Auto"), 0)
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("Tone Mapping"), tis.T("Intensity"), 
                                       ctypes.c_float(float(p_dict['Intensity'])))
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("Tone Mapping"), tis.T("Global Brightness Factor"), 
                                       ctypes.c_float(float(p_dict['GlobalBrightnessFactor'])))
    else:
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Tone Mapping"), tis.T("Enable"), 0) #WDR無効

    if 'Gamma' in p_dict:
        #Gamma: 0.1-5.0 default 1.0
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("Gamma"), tis.T("Value"),
                                       ctypes.c_float(float(p_dict['Gamma'])))

    if 'FPS' in p_dict:
        # fps: - 549
        ic.IC_SetFrameRate(hGrabber, ctypes.c_float(float(p_dict['FPS'])))

    if 'Exposure' in p_dict:
        # Exposure ：0.000001 - 30.0
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Exposure"), tis.T("Auto"), 0)
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("Exposure"), tis.T("Value"), 
                                       ctypes.c_float(float(p_dict['Exposure'])))
    else:
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Exposure"), tis.T("Auto"), 1) #Auto

    if 'Brightness' in p_dict:
        #Brightness : 0 - 4095 Default 240
        ic.IC_SetPropertyValue(hGrabber, tis.T("Brightness"), tis.T("Value"),
                               ctypes.c_int(int(p_dict['Brightness'])))

    if 'Gain' in p_dict:
        #Gain :0.0 - 48.0 Default 1.0
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Gain"), tis.T("Auto"), 0)
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("Gain"), tis.T("Value"), 
                                       ctypes.c_float(float(p_dict['Gain'])))
    else:
        ic.IC_SetPropertySwitch(hGrabber, tis.T("Gain"), tis.T("Auto"), 1) #Auto

    if 'WhiteBalanceRed' in p_dict and 'WhiteBalanceGreen' in p_dict and 'WhiteBalanceBlue' in p_dict:
        #WhiteBalance ： 各色 0.0 - 3.984375 ※IC Captureなどで実写を見て調整
        ic.IC_SetPropertySwitch(hGrabber, tis.T("WhiteBalance"), tis.T("Auto"), 0)
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("WhiteBalance"), tis.T("White Balance Red"), 
                                       ctypes.c_float(float(p_dict['WhiteBalanceRed'])))
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("WhiteBalance"), tis.T("White Balance Green"), 
                                       ctypes.c_float(float(p_dict['WhiteBalanceGreen'])))              
        ic.IC_SetPropertyAbsoluteValue(hGrabber, tis.T("WhiteBalance"), tis.T("White Balance Blue"), 
                                       ctypes.c_float(float(p_dict['WhiteBalanceBlue'])))
    else:
        ic.IC_SetPropertySwitch(hGrabber, tis.T("WhiteBalance"), tis.T("Auto"), 1) #Auto
    return

class LoadT4TISCams:
    # Tile
    def __init__(self, sources='T4TISCams.txt', img_size=640, stride=32, auto=True):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.img_size = img_size
        self.stride = stride
        self.flag = True
        self.rbt_flag = False # デバイスロストなどで自動的に自分を止める（再起動要否の目印）フラグ
        self.bubun = 40 # 前と新しい画像の比較に使う四角形部分の一辺のピクセル数 ★必ず偶数にすること！！！
        self.bad_cam = "" # デバイスロストしたカメラの位置情報を渡す変数

        if os.path.isfile(sources):
            with codecs.open(sources, 'r', 'utf-8') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        try:
            # TISカメラのためにimportする
            import ctypes
            import tisgrabber as tis
        except:
            print('tisgrabber.py,　tisgrabber_x64.dll など必要なファイルがありません。 ご確認ください!')
            sys.exit(0)
        self.imgs, self.frames, self.threads = [None] * 4, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto

        self.cnt = 0 # maenoとnowの同一画像検出の回数カウンタ
        self.maeno = [None] * 4 # 比較用画像を保存する変数
        self.now = [None] * 4
        self.camset = [None] * 4 # カメラ個別設定用辞書を読込む変数。初期はNoneとして判断に使う。

        self.fps = 70
        self.w = 640
        self.h = 480 # temporary definition
        vformat = "RGB24 ({0}x{1})".format(self.w, self.h) # カメラのビデオフォーマットを指定する定数
        self.obi = np.full((20, self.w, 3), (255, 255, 255), dtype=np.uint8) # 動画情報を表示するための帯
        self.top_obi = self.obi.copy()
        self.top_obi[0:20, 0:20] = (0, 0, 255) # 上のオビは左端を赤にしてプログラム停止のクリックの目印とする。
        for i, s in enumerate(sources):  # index, source
            # 初めに画像比較用の前の画像に当たるものを用意しておく
            self.now[i] = np.full((self.bubun, self.bubun, 3), (0, 0, 255), dtype=np.uint8)
            self.maeno[i] = np.full((self.bubun, self.bubun, 3), (0, 255, 0), dtype=np.uint8)
            # カメラの立上り順によるエラーを回避するために予め赤色の画面をカメラの数だけ用意しておく
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)
            # カメラ立上げのループの前に設定ファイルの有無を確認して、あらかじめ取込んでおく
            sn = s.split()[-1] # 'DFK 37BUX287 11223344' を分割して最後のS/Nのみ取り出し
            if os.path.exists(sn + '.txt'): # 個別のパラメータ設定ファイルがあったら
                print(f'{i}番カメラ：S/N{sn}　の設定テキストファイルが見つかりました。個別設定します。', end = '')
                self.camset[i] = get_camera_params(sn)
                print(f'{i} Done !')

        self.ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll") # TISおまじない1
        tis.declareFunctions(self.ic) # TISおまじない2
        self.ic.IC_InitLibrary(0) # TISおまじない3
        self.hGrabber = [None] * 4 # カメラインスタンスを格納するリストを定義しておく

        for i, s in enumerate(sources):  # index, source
            self.frames[i] = float('inf')  # infinite stream fallback
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = str(s)
            self.hGrabber[i] = self.ic.IC_CreateGrabber()
            self.ic.IC_OpenDevByUniqueName(self.hGrabber[i], tis.T(s)) # シリアルナンバーの指定も可能
            self.ic.IC_SetVideoFormat(self.hGrabber[i], tis.T(vformat))
            if (self.ic.IC_IsDevValid(self.hGrabber[i])): # カメラが開けたら
                if self.camset[i] != None: # 個別のパラメータ設定がNoneでなければ
                    p_dict = self.camset[i]
                else:
                    # 個別に指定しないときのデフォルトパラメータ指定 IC Captureなどで実写を見て調整
                    p_dict = {'FPS': '80',
                              'Exposure': '0.004',
                              'Brightness': '240',
                              'Gain': '0.0',
                              'WhiteBalanceRed': '1.66',
                              'WhiteBalanceGreen': '1.00', 
                              'WhiteBalanceBlue': '2.48',
                              #'Gamma': '0.7',
                              #'Intensity': '0.5', 
                              #'GlobalBrightnessFactor': '0.0',
                              #'ex':'使わないところは#でコメントアウト可能',
                             }
                set_camera_params(p_dict, i, self.hGrabber[i], self.ic, ctypes, tis)
                
                # Start the live video stream, but show no own live video window. We will use OpenCV for this.
                self.ic.IC_StartLive(self.hGrabber[i], 0) # 引数を「１」にするとライブ画像が開く。OpenCVでの描画をするので「０」とする。
                #print('★★ic.IC_SnapImage(hGrabber[',i, ']: ', ic.IC_SnapImage(hGrabber[i])) #debugprint

                # 連続取り込みのスレッドを起動する
                self.threads[i] = Thread(target=self.update, args=([i, self.hGrabber[i], s, self.ic, ctypes, tis]), daemon=True)
                self.threads[i].start()
                print(f"{st} Success ({self.frames[i]} frames {self.w}x{self.h} at {float(p_dict['FPS']):.2f} FPS)")

            else: # カメラが開けない時
                print(f'{st}Failed to open Cam {s}')
                self.ic.IC_CloseVideoCaptureDevice(self.hGrabber[i])
                self.ic.IC_ReleaseGrabber(self.hGrabber[i]) 
        self.rect = True  # dummy code. rect inference if all shapes equal

    def update(self, i, hGrabber, stream, ic, ctypes, tis):
        Width = ctypes.c_long()
        Height = ctypes.c_long()
        BitsPerPixel = ctypes.c_int()
        colorformat = ctypes.c_int()
        cnt_a = 0 # 画像が取込めなかった連続回数のカウンタ
        while (self.ic.IC_IsDevValid(hGrabber)) and self.flag:
            # かなり長い記述になるが以下self.imgs[i] = im までで画像をOpenCVに渡せる形で取得している
            if self.ic.IC_SnapImage(hGrabber) == tis.IC_SUCCESS:
                # Query values of image description
                self.ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel, colorformat)
                # Calculate the buffer size
                bpp = int(BitsPerPixel.value / 8.0)
                buffer_size = Width.value * Height.value * BitsPerPixel.value
                imagePtr = self.ic.IC_GetImagePtr(hGrabber)
                imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
                # Create the numpy array
                im = np.ndarray(buffer=imagedata.contents, dtype=np.uint8, shape=(Height.value, Width.value, bpp))
                im = cv2.flip(im, 0)
                self.imgs[i] = im
                cnt_a = 0
                #time.sleep(1 / self.fps)  # wait timeはTISカメラでは不要

            else: # 画像が上手く取り込めなかったときの処理。
                # 産業用カメラでも必ず画像の取りこぼしが起きるので一度や二度で止めてはいけない。ここでは10回連続で異常と判断する。
                print(f'WARNING: Cam{i} 画像が正常に取込めていません。')
                cnt_a += 1
                self.imgs[i] =  np.full((self.h, self.w, 3), (98, 244, 255), dtype=np.uint8) # 黄色い画像にする
                if cnt_a >= 10: # 画像が正常に取り込めない状態が10回続いたらループを抜ける
                    print(f'Cam{i} 画像が取込めない状態が{cnt_a}ループ続いたのでループから抜けます。')
                    break

        # 何らかの理由でループを抜けてしまった場合もブルーバック画像とする。ここに来るのはEscで意識的に止めた時とic.IC_IsDevValid(hGrabber)がFalseの時+ic.IC_SnapImage(hGrabber) == tis.IC_SUCCESSでない時。
        self.imgs[i] =  np.full((self.h, self.w, 3), (255, 0, 0), dtype=np.uint8)        
        self.ic.IC_StopLive(hGrabber)
        self.ic.IC_CloseVideoCaptureDevice(hGrabber)
        self.ic.IC_ReleaseGrabber(hGrabber)        
        print('画像取込のループを抜けました。 Cam:', i)
        return

    def __iter__(self):
        return self

    def __next__(self):
        ky = cv2.waitKey(1)
        if ky == 27 or self.rbt_flag: # esc to quit 
            self.flag = False
            cv2.destroyAllWindows()
            if ky == 27:
                print('キー入力により停止しました。')
            #time.sleep(1) # カメラスレッドの終了待ち
            self.ic.IC_CloseLibrary()
            raise StopIteration
        if not self.flag:
            self.ic.IC_CloseLibrary()
            raise StopIteration

        self.now[0] = self.imgs[0][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[1] = self.imgs[1][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]   
        self.now[2] = self.imgs[2][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[3] = self.imgs[3][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)] 

        if (self.now[0] == self.maeno[0]).all() or (self.now[1] == self.maeno[1]).all() or (self.now[2] == self.maeno[2]).all() or (self.now[3] == self.maeno[3]).all():
            self.cnt +=1
            if self.cnt >= self.fps * 4 : # 画像が更新されないという判断が数秒続いたら…
                self.flag = False
                if (self.now[0] == self.maeno[0]).all():
                    self.bad_cam = "左上"
                elif (self.now[1] == self.maeno[1]).all():
                    self.bad_cam = "右上"
                elif (self.now[2] == self.maeno[2]).all():
                    self.bad_cam = "右下"
                elif (self.now[3] == self.maeno[3]).all():
                    self.bad_cam = "左下"
                self.rbt_flag = True # 終了後、自分を再起動するフラグを立てる
        else:
            self.cnt = 0 # 比較結果が異なればカウンタをリセット

        # ここで4つの画像を合成する
        self.concimg = cv2.hconcat([self.imgs[0], self.imgs[1]])
        conc2 = cv2.hconcat([self.imgs[3], self.imgs[2]])
        self.concimg = cv2.vconcat([self.concimg, conc2])
        self.concimg = cv2.resize(self.concimg, (self.w, self.h), interpolation = cv2.INTER_AREA)
        self.concimg = cv2.vconcat([self.top_obi, self.concimg, self.obi])
        #self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 

        self.maeno[0] = self.now[0] # 比較用画像の入れ替え
        self.maeno[1] = self.now[1] # 比較用画像の入れ替え
        self.maeno[2] = self.now[2] # 比較用画像の入れ替え
        self.maeno[3] = self.now[3] # 比較用画像の入れ替え

        img0 = self.concimg.copy()
        # Letterbox
        #img_lb = letterbox(img0)[0] # letterbox関数から返ってきた画像部分のみ
        img_lb = None # 202409現在letterbox処理はMultiBackendに任せるので…

        return self.sources, img_lb, img0, self.rbt_flag, self.bad_cam

class LoadV4TISCams:
    # Vertical
    def __init__(self, sources='V4TISCams.txt', img_size=640, stride=32, auto=True):
        torch.backends.cudnn.benchmark = True  # faster for fixed-size inference 最新のstream_loader.pyから登用
        self.img_size = img_size
        self.stride = stride
        self.flag = True
        self.rbt_flag = False # デバイスロストなどで自動的に自分を止める（再起動要否の目印）フラグ
        self.bubun = 40 # 前と新しい画像の比較に使う四角形部分の一辺のピクセル数 ★必ず偶数にすること！！！
        self.bad_cam = "" # デバイスロストしたカメラの位置情報を渡す変数

        if os.path.isfile(sources):
            with codecs.open(sources, 'r', 'utf-8') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        try:
            # TISカメラのためにimportする
            import ctypes
            import tisgrabber as tis
        except:
            print('tisgrabber.py,　tisgrabber_x64.dll など必要なファイルがありません。 ご確認ください!')
            sys.exit(0)
        self.imgs, self.frames, self.threads = [None] * 4, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto

        self.cnt = 0 # maenoとnowの同一画像検出の回数カウンタ
        self.maeno = [None] * 4 # 比較用画像を保存する変数
        self.now = [None] * 4
        self.camset = [None] * 4 # カメラ個別設定用辞書を読込む変数。初期はNoneとして判断に使う。

        self.fps = 80
        self.w = 720 #640
        self.h = 180 #160
        vformat = "RGB24 ({0}x{1})".format(self.w, self.h) # カメラのビデオフォーマット
        self.obi = np.full((20, self.w, 3), (255, 255, 255), dtype=np.uint8) # 動画情報を表示するための帯
        self.top_obi = self.obi.copy()
        self.top_obi[0:20, 0:20] = (0, 0, 255) # 上のオビは左端を赤にしてプログラム停止のクリックの目印とする。
        for i, s in enumerate(sources):
            # 初めに画像比較用の前の画像に当たるものを用意しておく
            self.now[i] = np.full((self.bubun, self.bubun, 3), (0, 0, 255), dtype=np.uint8)
            self.maeno[i] = np.full((self.bubun, self.bubun, 3), (0, 255, 0), dtype=np.uint8)
            # カメラの立上り順によるエラーを回避するために予め赤色の画面をカメラの数だけ用意しておく
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)
            # カメラ立上げのループの前に設定ファイルの有無を確認して、あらかじめ取込んでおく
            sn = s.split()[-1] # 'DFK 37BUX287 11223344' を分割して最後のS/Nのみ取り出し
            if os.path.exists(sn + '.txt'): # 個別のパラメータ設定ファイルがあったら
                print(f'{i}番カメラ：S/N{sn}　の設定テキストファイルが見つかりました。個別設定します。', end = '')
                self.camset[i] = get_camera_params(sn)
                print(f'{i} Done !')

        self.ic = ctypes.cdll.LoadLibrary("./tisgrabber_x64.dll") # TISおまじない1
        tis.declareFunctions(self.ic) # TISおまじない2
        self.ic.IC_InitLibrary(0) # TISおまじない3
        self.hGrabber = [None] * 4 # カメラインスタンスを格納するリストを定義しておく

        for i, s in enumerate(sources):  # index, source
            self.frames[i] = float('inf')  # infinite stream fallback
            # Start thread to read frames from video stream
            st = f'Cam {i}: {s}... '
            s = str(s)
            self.hGrabber[i] = self.ic.IC_CreateGrabber()
            self.ic.IC_OpenDevByUniqueName(self.hGrabber[i], tis.T(s)) # シリアルナンバーの指定も可能
            self.ic.IC_SetVideoFormat(self.hGrabber[i], tis.T(vformat))
            if (self.ic.IC_IsDevValid(self.hGrabber[i])): # カメラが開けたら
                if self.camset[i] != None: # 個別のパラメータ設定がNoneでなければ
                    p_dict = self.camset[i]
                else:
                    # 個別に指定しないときのパラメータ指定 IC Captureなどで実写を見て調整
                    p_dict = {'FPS': '80',
                              'Exposure': '0.004',
                              'Brightness': '240',
                              'Gain': '25.0',
                              'WhiteBalanceRed': '1.66',
                              'WhiteBalanceGreen': '1.00', 
                              'WhiteBalanceBlue': '2.48',
                              'Gamma': '0.7',
                              'Intensity': '0.5', 
                              'GlobalBrightnessFactor': '0.0',
                              #'ex':'使わないところは#でコメントアウト可能',
                             }
                set_camera_params(p_dict, i, self.hGrabber[i], self.ic, ctypes, tis)

                # Start the live video stream, but show no own live video window. We will use OpenCV for this.
                self.ic.IC_StartLive(self.hGrabber[i], 0) # 引数を「１」にするとライブ画像が開く。OpenCVでの描画をするので「０」とする。
                #print('★★ic.IC_SnapImage(hGrabber[',i, ']: ', ic.IC_SnapImage(hGrabber[i])) #debugprint

                # 連続取り込みのスレッドを起動する
                self.threads[i] = Thread(target=self.update, args=([i, self.hGrabber[i], s, self.ic, ctypes, tis]), daemon=True)
                self.threads[i].start()
                print(f"{st} Success ({self.frames[i]} frames {self.w}x{self.h} at {float(p_dict['FPS']):.2f} FPS)")

            else: # カメラが開けない時
                print(f'{st}Failed to open Cam {s}')
                self.ic.IC_CloseVideoCaptureDevice(self.hGrabber[i])
                self.ic.IC_ReleaseGrabber(self.hGrabber[i])
        self.rect = True  # dummy code. rect inference if all shapes equal

    def update(self, i, hGrabber, stream, ic, ctypes, tis):
        Width = ctypes.c_long()
        Height = ctypes.c_long()
        BitsPerPixel = ctypes.c_int()
        colorformat = ctypes.c_int()
        cnt_a = 0 # 画像が取込めなかった連続回数のカウンタ
        while (self.ic.IC_IsDevValid(hGrabber)) and self.flag:
            # かなり長い記述になるが以下self.imgs[i] = im までで画像をOpenCVに渡せる形で取得している
            if self.ic.IC_SnapImage(hGrabber) == tis.IC_SUCCESS:
                # Query values of image description
                self.ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel, colorformat)
                # Calculate the buffer size
                bpp = int(BitsPerPixel.value / 8.0)
                buffer_size = Width.value * Height.value * BitsPerPixel.value
                imagePtr = self.ic.IC_GetImagePtr(hGrabber)
                imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))
                # Create the numpy array
                im = np.ndarray(buffer=imagedata.contents, dtype=np.uint8, shape=(Height.value, Width.value, bpp))
                im = cv2.flip(im, 0)
                self.imgs[i] = im
                cnt_a = 0
                #time.sleep(1 / self.fps)  # wait timeはTISカメラでは不要

            else: # 画像が上手く取り込めなかったときの処理。
                # 産業用カメラでも必ず画像の取りこぼしが起きるので一度や二度で止めてはいけない。ここでは10回連続で異常と判断する。
                print(f'WARNING: Cam{i} 画像が正常に取込めていません。')
                cnt_a += 1
                self.imgs[i] =  np.full((self.h, self.w, 3), (98, 244, 255), dtype=np.uint8) # 黄色い画像にする
                if cnt_a >= 10: # 画像が正常に取り込めない状態が10回続いたらループを抜ける
                    print(f'Cam{i} 画像が取込めない状態が{cnt_a}ループ続いたのでループから抜けます。')
                    break

        # 何らかの理由でループを抜けてしまった場合もブルーバック画像とする。ここに来るのはEscで意識的に止めた時とic.IC_IsDevValid(hGrabber)がFalseの時+ic.IC_SnapImage(hGrabber) == tis.IC_SUCCESSでない時。
        self.imgs[i] =  np.full((self.h, self.w, 3), (255, 0, 0), dtype=np.uint8) # 青い画像にする
        self.ic.IC_StopLive(hGrabber)
        self.ic.IC_CloseVideoCaptureDevice(hGrabber)
        self.ic.IC_ReleaseGrabber(hGrabber)
        print('画像取込のループを抜けました。 Cam:', i)
        return

    def __iter__(self):
        return self

    def __next__(self):
        ky = cv2.waitKey(1)
        if ky == 27 or self.rbt_flag: # esc to quit 
            self.flag = False
            cv2.destroyAllWindows()
            if ky == 27:
                print('キー入力により停止しました。')
            #time.sleep(1) # カメラスレッドの終了待ち
            self.ic.IC_CloseLibrary()
            raise StopIteration
        if not self.flag:
            self.ic.IC_CloseLibrary()
            raise StopIteration

        # 比較用画像の切り出し
        self.now[0] = self.imgs[0][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[1] = self.imgs[1][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]   
        self.now[2] = self.imgs[2][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)]
        self.now[3] = self.imgs[3][int(self.h/2) - int(self.bubun/2):int(self.h/2) + int(self.bubun/2), int(self.w/2) - int(self.bubun/2):int(self.w/2) + int(self.bubun/2)] 
        
        if (self.now[0] == self.maeno[0]).all() or (self.now[1] == self.maeno[1]).all() or (self.now[2] == self.maeno[2]).all() or (self.now[3] == self.maeno[3]).all():
            self.cnt +=1
            if self.cnt >= self.fps * 4 : # 画像が更新されないという判断が数秒続いたら…
                self.flag = False
                if (self.now[0] == self.maeno[0]).all():
                    self.bad_cam = "一番上"
                elif (self.now[1] == self.maeno[1]).all():
                    self.bad_cam = "二番目"
                elif (self.now[2] == self.maeno[2]).all():
                    self.bad_cam = "三番目"
                elif (self.now[3] == self.maeno[3]).all():
                    self.bad_cam = "一番下"
                self.rbt_flag = True # 終了後、自分を再起動するフラグを立てる
        else:
            self.cnt = 0 # 比較結果が異なればカウンタをリセット

        # ここで4つの画像を合成する
        self.concimg = cv2.vconcat([self.top_obi, self.imgs[0], self.imgs[1],self.imgs[2], self.imgs[3], self.obi]) # 縦積みなら一度でvconcat出来る。

        self.maeno[0] = self.now[0] # 比較用画像の入れ替え
        self.maeno[1] = self.now[1] # 比較用画像の入れ替え
        self.maeno[2] = self.now[2] # 比較用画像の入れ替え
        self.maeno[3] = self.now[3] # 比較用画像の入れ替え

        img0 = self.concimg.copy()
        # Letterbox
        #img_lb = letterbox(img0)[0] # letterbox関数から返ってきた画像部分のみ
        img_lb = None # 202409現在letterbox処理はMultiBackendに任せるので…

        return self.sources, img_lb, img0, self.rbt_flag, self.bad_cam

class LoadT4Streams:
    # for USB camera  Tile
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        global flag
        self.img_size = img_size
        self.stride = stride
        self.flag = True # 複数開いたカメラスレッドを閉じるためのフラグ
        self.rbt_flag = False # デバイスロストなどで自動的に自分を止める（再起動要否の目印）フラグ
        self.bad_cam = "" # デバイスロストしたカメラの位置情報を渡す変数
        self.w = 640
        self.h = 480

        if os.path.isfile(sources):
            with codecs.open(sources, 'r', 'utf-8') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * 4, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        self.obi = np.full((20, self.w, 3), (255, 255, 255), dtype=np.uint8) # 動画情報を表示するための帯
        self.top_obi = self.obi.copy()
        self.top_obi[0:20, 0:20] = (0, 0, 255) # 上のオビは左端を赤にしてプログラム停止のクリックの目印とする。
        
        # カメラの立ち上がり方次第でエラーを起こすことあるので、予め赤色の画面をカメラの数だけ用意しておく
        for i, s in enumerate(sources):  # index, source
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)

        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s + cv2.CAP_DSHOW)
            #assert cap.isOpened(), f'{st}Failed to open {s}'
            w = self.w #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = self.h #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            if cap.isOpened():
                _, self.imgs[i] = cap.read()  # guarantee first frame
                self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=False)
                # threadsは、daemon=Trueで複数起動すると終了時にカメラを開放しなくなる。そのためdaemon=False（デフォ）とした。
                print(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
                self.threads[i].start()
                #print('** ', self.threads) # debug print
            else:
                print(f'{st}Failed to open Cam {s}')
                self.imgs[i] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)

        print('')  # newline

        self.rect = True #np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f and self.flag: # flagもループの条件に加えている
            start_t = time.perf_counter()
            n += 1
            #_, self.imgs[i] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    print('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            end_t = time.perf_counter()
            #print(str(i) + '　elapse time = {:.3f} Seconds'.format((end_t - start_t))) 
            time.sleep(1 / self.fps[i])  # wait time
        cap.release() # 無限ループから抜けたらカメラインスタンスを開放するのを忘れないこと！

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        #if not all(x.isAlive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
        if cv2.waitKey(1) == 27:  # esc to quit 
            self.flag = False # 画像取込の無限ループを抜けるためフラグを書き換える
            cv2.destroyAllWindows()          
            raise StopIteration

        #h, w, _ = self.imgs[0].shape # 画像のサイズを取込んでおく
        # ここで4つの画像を合成する
        if len(self.sources) == 1:
            self.imgs[1] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
        self.concimg = cv2.hconcat([self.imgs[0], self.imgs[1]])
        if len(self.sources) > 2:
            if len(self.sources) == 3:
                self.imgs[3] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
            conc2 = cv2.hconcat([self.imgs[2], self.imgs[3]])
            self.concimg = cv2.vconcat([self.concimg, conc2])
            self.concimg = cv2.resize(self.concimg, (800, 600), interpolation = cv2.INTER_AREA)
        else:
            self.concimg = cv2.resize(self.concimg, (800, 300), interpolation = cv2.INTER_AREA)
        self.concimg = cv2.resize(self.concimg, (self.w, self.h), interpolation = cv2.INTER_AREA)
        self.concimg = cv2.vconcat([self.top_obi, self.concimg, self.obi])
        #self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 

        img0 = self.concimg.copy()
        # Letterbox
        #img_lb = letterbox(img0)[0] # letterbox関数から返ってきた画像部分のみ
        img_lb = None # 202409現在letterbox処理はMultiBackendに任せるので…

        return self.sources, img_lb, img0, self.rbt_flag, self.bad_cam

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years

class LoadV4Streams:
    # for USB camera  Vertical
    def __init__(self, sources='Vstreams.txt', img_size=640, stride=32, auto=True):
        global flag
        self.img_size = img_size
        self.stride = stride
        self.flag = True # 複数開いたカメラスレッドを閉じるためのフラグ
        self.rbt_flag = False # デバイスロストなどで自動的に自分を止める（再起動要否の目印）フラグ
        self.bad_cam = "" # デバイスロストしたカメラの位置情報を渡す変数

        if os.path.isfile(sources):
            with codecs.open(sources, 'r', 'utf-8') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip()) and x[0] != '#']
        else:
            sources = [sources]

        print(sources)
        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * 4, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        self.w = 640 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = 160 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_h = 480 # クロップしない場合の縦画素数
        self.start_h = int((full_h - self.h) / 2)
        self.obi = np.full((20, self.w, 3), (255, 255, 255), dtype=np.uint8) # 動画情報を表示するための帯
        self.top_obi = self.obi.copy()
        self.top_obi[0:20, 0:20] = (0, 0, 255) # 上のオビは左端を赤にしてプログラム停止のクリックの目印とする。
        
        # 予め赤色の画面をカメラの数だけ用意しておく
        for i, s in enumerate(sources):  # index, source
            self.imgs[i] = np.full((self.h, self.w, 3), (0, 0, 255), dtype=np.uint8)

        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s + cv2.CAP_DSHOW)
            #assert cap.isOpened(), f'{st}Failed to open {s}'

            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            if cap.isOpened():
                _, im = cap.read()  # guarantee first frame
                self.imgs[i] = im[self.start_h:(self.start_h + self.h), 0:self.w] # crop
                self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=False)
                # threadsは、daemon=Trueで複数起動すると終了時にカメラを開放しなくなる。そのためdaemon=False（デフォ）とした。
                print(f"{st} Success ({self.frames[i]} frames {self.w}x{self.h} at {self.fps[i]:.2f} FPS)")
                self.threads[i].start()
                #print('** ', self.threads) # debug print
            else:
                print(f'{st}Failed to open Cam {s}')
                self.imgs[i] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)

        print('')  # newline
        
        self.rect = True #np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f and self.flag: # flagもループの条件に加えている
            n += 1
            #_, self.imgs[i] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:

                    self.imgs[i] = im[self.start_h:(self.start_h + self.h), 0:self.w] # 取り込んだ画像の高さ方向で中心部分だけを使う
                else:
                    print('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time
        cap.release() # 無限ループから抜けたらカメラインスタンスを開放するのを忘れないこと！

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        #if not all(x.isAlive() for x in self.threads) or cv2.waitKey(1) == 27: #ord('q'):  # q to quit
        if cv2.waitKey(1) == 27: #ord('q'):  # q to quit 
            self.flag = False # 画像取込の無限ループを抜けるためフラグを書き換える
            cv2.destroyAllWindows()          
            raise StopIteration

        h, w, _ = self.imgs[0].shape # 画像のサイズを取込んでおく
        # ここで4つの画像を合成する
        if len(self.sources) == 1:
            self.imgs[1] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.imgs[0], self.imgs[1]])
        if len(self.sources) > 2:
            if len(self.sources) == 3:
                self.imgs[3] = np.full((self.h, self.w, 3), (128, 128, 128), dtype=np.uint8)
            conc2 = cv2.vconcat([self.imgs[2], self.imgs[3]])
            self.concimg = cv2.vconcat([self.concimg, conc2])
            self.concimg = cv2.resize(self.concimg, (self.w, 4*self.h), interpolation = cv2.INTER_AREA)
        else:
            self.concimg = cv2.resize(self.concimg, (self.w, 2*self.h), interpolation = cv2.INTER_AREA)
        self.obi = np.full((20, self.w, 3), (255, 255, 255), dtype=np.uint8)
        self.concimg = cv2.vconcat([self.top_obi, self.concimg, self.obi])
        #self.concimg = np.expand_dims(self.concimg, axis=0) # CHW > BCHW 

        img0 = self.concimg.copy()
        # Letterbox
        #img_lb = letterbox(img0)[0] # letterbox関数から返ってきた画像部分のみ
        img_lb = None # 202409現在letterbox処理はMultiBackendに任せるので…

        return self.sources, img_lb, img0, self.rbt_flag, self.bad_cam

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years

