# -*- coding: utf-8 -*-
"""
@file      :  thread_test.py
@Time      :  2022/8/28 14:18
@Software  :  PyCharm
@summary   :  Test_multiple_epochs
@Author    :  Bajian Xiang
"""

import datetime
import os
import threading

def execCmd(cmd):
    try:
        print("COMMAND -- %s -- BEGINS -- %s -- " % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("COMMAND -- %s -- ENDS -- %s -- " % (cmd, datetime.datetime.now()))
    except:
        print("Failed -- %s -- " % cmd)

if __name__ == "__main__":
    epochs = [15, 21, 34]
    is_parallel = True  # 是否并行

    if not is_parallel:
        # 串行
        for i in range(len(epochs)):
            command = "python classification_test_resnet50.py --epoch_for_save " + str(epochs[i]) + " --outputresult_dir " + "./0828_new_test/epoch" + str(epochs[i])
            os.system(command)
    else:
        # 并行
        commands = ["python test_FPN_STI.py --epoch_for_save " + str(epochs[i]) + " --outputresult_dir " + "./0923_test/epoch" + str(epochs[i]) for i in range(len(epochs))]
        threads = []
        for cmd in commands:
            th = threading.Thread(target=execCmd, args=(cmd,))
            th.start()
            threads.append(th)
        # 等待线程运行完毕
        for th in threads:
            th.join()
