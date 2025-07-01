# Elden Ring Route Planner
一个基于Python和OpenCV的辅助工具，用于识别艾尔登法环黑夜君临地图上的据点，并使用TSP/点插入算法规划出一条高效的探索路线。

## 主要功能
- 使用多模板匹配技术，精确识别多种地图图标。
- 采用最近邻算法/点插入算法快速生成路线。
- 结果可视化，在地图上清晰地标出路线和顺序。

## 如何使用
1. 确保已安装Python和必要的库 (`opencv-python`, `numpy`, `matplotlib`)。
2. 修改脚本中的文件路径 (`MAP_IMAGE_PATH`, `TEMPLATE_FOLDER`)。
3. 运行Jupyter Notebook或Python脚本。

## 未来计划
- [ ] 封装成独立的.exe应用程序。
- [ ] 实现实时屏幕捕捉功能。