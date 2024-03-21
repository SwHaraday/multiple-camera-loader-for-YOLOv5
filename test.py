import argparse
import time, datetime
import cv2
import numpy as np
import os, sys
import subprocess
from cam_loader import LoadT4TISCams, LoadV4TISCams

#★★★★　ここから、次の★★★★までの間を用途・環境に合わせて書き換えて下さい。★★★★
# 画面に表示する倍率
DisplayScale = 1.0

#★★★★ ここまでの間を用途・環境に合わせて書き換えて下さい。★★★★

def run(source= 'sources.txt',
        ): 
    # 引数 --source で指定されたファイル名に応じてcam_loader.pyのクラスを呼び出す
    if source == 'sources.txt': # 通常のUSBカメラ複数使用の場合
        cams = LoadT4TISCams(source)
        src_name = '4 Tile TIS cams '

    elif source == 'sources_V.txt': # TIS社の産業用カメラの場合
        cams = LoadV4TISCams(source)
        src_name = '4 Vertical TIS cams '

    for sources, frame_lb, frame, rbt_flag, bad in cams:
        h, w, _ = frame.shape # 画像のサイズ取り込み
        # 例えばの話このあたりにAIの処理などを挟んでみる
        frame_disp = cv2.resize(frame, dsize=None, fx=DisplayScale, fy=DisplayScale)
        #--- 描画した画像を表示
        cv2.imshow('Cameras from 4direction -source ' + src_name + '  **Hit "q" to stop', frame_disp)
        k = cv2.waitKey(1)
        if k == ord('r') or cams.rbt_flag:
            global rbt, bad_cam
            rbt = True
            bad_cam = cams.bad_cam
            cams.flag = False #インスタンス化した画像取り込みプログラムに停止の合図を送る
            cv2.destroyAllWindows()
            break
    time.sleep(3) #　数秒スリープしてthreadを先に終了させる

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default= '4TISCams.txt', help='file/dir/URL/glob, 0 for webcam')
    #parser.add_argument('--dummy',action='store_true', help='指定すれば開けないカメラ部分にダミー画像を使う。')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    rbt = False
    t = datetime.datetime.now()
    print("★　{}".format(t.strftime("%Y/%m/%d %H:%M:%S")) + " Started...")
    opt = parse_opt()
    main(opt)
    t = datetime.datetime.now()
    print("★★　{}".format(t.strftime("%Y/%m/%d %H:%M:%S")) + " Stopped...")
    if rbt:
        try:
            with open('reboot.log', 'a') as f: # logファイルを追記モードで開く
                tm = "{} ".format(t.strftime("%Y/%m/%d %H:%M:%S"))
                tm = tm + bad_cam + " カメラのデバイスロストを検出し自動再起動しました。"
                print(tm, file=f) # 日時とカメラ位置をファイルに書き込む
        except:
            print("再起動ログが書き込めませんでした。reboot.txtの有無を確認してください。")
        print(f'{bad_cam} カメラのデバイスロストを検出したので自動再起動しました。')
        res = subprocess.Popen("rbt.bat", creationflags=subprocess.CREATE_NEW_CONSOLE)     