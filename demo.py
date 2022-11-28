from track import *
import tempfile
import cv2
import torch
import streamlit as st
import os


if __name__ == '__main__':
    st.title('Vehicle detection and counting')
    st.markdown('<h3 style="color: red"> with Yolov5 and Deep SORT </h3', unsafe_allow_html=True)

    # upload video
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

    if video_file_buffer:
        st.sidebar.text('Input video')
        st.sidebar.video(video_file_buffer)
        # save video from streamlit into "videos" folder for future detect
        with open(os.path.join('videos', video_file_buffer.name), 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    st.sidebar.markdown('---')
    st.sidebar.title('Settings')

    # setting hyperparameter
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    line = st.sidebar.number_input('Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    st.sidebar.markdown('---')

    custom_class = st.sidebar.checkbox('Custom classes')
    assigned_class_id = []
    names = ['car', 'motorcycle','bus', 'truck']

    # custom classes
    if custom_class:
        assigned_class = st.sidebar.multiselect('Select custom classes', list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))
    

    # st.write(assigned_class_id)
    
    status = st.empty()
    stframe = st.empty()
    if video_file_buffer is None:
        status.markdown('<font size= "4"> **Status:** Waiting for input </font>', unsafe_allow_html=True)
    else:
        status.markdown('<font size= "4"> **Status:** Ready </font>', unsafe_allow_html=True)

    car, bus, truck, motor = st.columns(4)
    with car:
        st.markdown('**Car**')
        car_text = st.markdown('__')
    
    with bus:
        st.markdown('**Bus**')
        bus_text = st.markdown('__')

    with truck:
        st.markdown('**Truck**')
        truck_text = st.markdown('__')
    
    with motor:
        st.markdown('**Motorcycle**')
        motor_text = st.markdown('__')

    fps, _,  _, _  = st.columns(4)
    with fps:
        st.markdown('**FPS**')
        fps_text = st.markdown('__')


    track_button = st.sidebar.button('START')
    reset_button = st.sidebar.button('RESET ID')
    if track_button:
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = f'videos/{video_file_buffer.name}'

        status.markdown('<font size= "4"> **Status:** Running... </font>', unsafe_allow_html=True)
        with torch.no_grad():
            detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text)
        status.markdown('<font size= "4"> **Status:** Finished ! </font>', unsafe_allow_html=True)
        # end_noti = st.markdown('<center style="color: blue"> FINISH </center>',  unsafe_allow_html=True)

    # reset ID and count from 0
    if reset_button:
        reset()
        st.markdown('<h3 style="color: blue"> Reseted ID </h3>', unsafe_allow_html=True)
