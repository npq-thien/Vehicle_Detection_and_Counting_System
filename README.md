# Vehicle Detection and Counting System on Streamlit

## Introduction
This project is used to count and detect vehicle on the highway. It can detect 4 types of vehicles: car, motorcycle, bus, truck.
I run this project on Python 3.9.7

Input: video size should smaller than 200 MB

Output: number of each vehicle type
## What we use
* [YOLOv5](https://github.com/ultralytics/yolov5/releases) to detect objects on each of the video frames.

* [Deep SORT](https://github.com/nwojke/deep_sort) to track those objects over different frames and help counting.

* [Streamlit](https://github.com/streamlit/streamlit) to build a simple web.
## Installation

* Install essential libraries and packages:
```python
pip install -r requirements.txt
```

* Install Streamlit:
```python
pip install streamlit
```

* Run demo:
```python
streamlit run demo.py --server.maxUploadSize=500
```

**NOTE**: If the web keeps showing "Please wait...", try to install streamlit version 1.11.0
```python
pip install streamlit==1.11.0
```

# Demo features
1. Confidence: the confidence that one object belongs to one class
