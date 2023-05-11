# STI 训练

## STI训练代码路径

3块卡服务器：`/data/xbj/code/0406_awl_STI`

建议直接copy服务器文件夹，exclude save_model log *test *.log ，速度会快一些

## 运行说明

- 运行 `train_FPN_STI.py`即可进行训练，tensorboard和权重文件会分别存在log/和save_model/中
- parse_args中可以修改的参数
  - `save_dir`：权重存储的路径
  - `trained_epoch`：如果是restart，将其修改为之前训练的轮次
  - `load_pretrain`：如果是restart，将其修改为网络权重路径



# STI 测试及箱图绘制

## 运行说明

- 运行 `test_FPN_STI.py`，需要修改parse_args中的以下内容：
  - `model_path`：改为权重存储的【文件夹】路径
  - `epoch_for_save`：改为你需要test的具体epoch数
  - `outputresult_dir`：结果存储的文件夹路径
- 运行完之后，会在`outputresult_dir`中生成
  - 一个.pt文件
  - 一个.csv文件
  - 一个.png文件

## 绘制结果箱图

- 将cal_gen_boxplot.py复制到`outputresult_dir`下，并运行
  - 这个脚本服务器里也有，在`/data/xbj/code/0406_awl_STI/0409_test/epoch10`
- 运行结束后会生成一个`0409_all_Freq_without_outlier.png`，即为箱图
