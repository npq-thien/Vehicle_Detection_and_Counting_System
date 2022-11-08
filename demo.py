from track2 import *
import tempfile
import cv2
import torch
import streamlit as st

if __name__ == '__main__':
    st.title('Yolov5 with Deep SORT')

    st.sidebar.title('Settings')

    st.markdown(
    """
    <style>
    [data-testid ="stSidebar"][aria-expanded="true" > div:first-child{ width: 400px;}
    [data-testid ="stSidebar"][aria-expanded="false" > div:first-child{ width: 400px; margin-left: -400px}
    </style> 
    """,
    unsafe_allow_html=True
    )
    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')

    save_img = st.sidebar.checkbox('Save video')
    custom_class = st.sidebar.checkbox('Custom classes')
    assigned_class_id = []
    names = ['car', 'motorcycle','bus', 'truck']

    # custom classes
    if custom_class:
        assigned_class = st.sidebar.multiselect('Select custom classes', list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    # upload video
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])
    DEMO_VIDEO = 'test.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    # st.sidebar.text('Input video')
    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tffile.name - DEMO_VIDEO
        dem_video = open(tffile.name, 'rb')
        demo_bytes = dem_video.read()

        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)
    else:
        tffile.write(video_file_buffer.read())
        dem_video = open(tffile.name, 'rb')
        demo_bytes = dem_video.read()

        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)
    
    stframe = st.empty()
    st.sidebar.markdown('---')



    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')
    
    with kpi2:
        st.markdown('**Tracked object**')
        kpi1_text = st.markdown('0')

    with kpi3:
        st.markdown('**Frame Rate**')
        kpi1_text = st.markdown('0')

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='best_all.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='videos/motor.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    
    opt.source = f'videos/{video_file_buffer.name}'
    
    if video_file_buffer:
        tffile.write(video_file_buffer.read())
        dem_video = open(tffile.name, 'rb')
        demo_bytes = dem_video.read()

        st.video(demo_bytes)

    # vid.release()
    with torch.no_grad():
        detect(opt)
    
    # if source_index == 0:
    #     video_file_buffer = st.sidebar.file_uploader(
    #         "上传图片", type=['png', 'jpeg', 'jpg'])
    #     if video_file_buffer is not None:
    #         is_valid = True
    #         with st.spinner(text='资源加载中...'):
    #             st.sidebar.image(video_file_buffer)
    #             picture = Image.open(video_file_buffer)
    #             picture = picture.save(f'data/images/{video_file_buffer.name}')
    #             opt.source = f'data/images/{video_file_buffer.name}'
    #     else:
    #         is_valid = False
    # else:
    
    