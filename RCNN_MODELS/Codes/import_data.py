pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="RBz4SJR6hNiuVHyuaYLd")
project = rf.workspace("wastedataset-preprocessing").project("amchoproject")
version = project.version(3)
dataset = version.download("coco")
                