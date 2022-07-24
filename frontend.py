import streamlit as st
import os
import torch
from PIL import Image
import glob
import zipfile

def main():
    st.title('Test Predict YoloV5')
    imgs = st.file_uploader("Choose Images")
    
    if 'dcount' not in st.session_state:
        st.session_state['dcount'] = 0

    if not imgs:
        return
        
    with zipfile.ZipFile(imgs,"r") as zipf:
        st.session_state['dcount'] += 1
        zipf.extractall("dataset/v{}".format(st.session_state['dcount']))

    imgname = os.listdir("dataset/v{}".format(st.session_state['dcount']))
    preds = glob.glob("dataset/v{}/*.*".format(st.session_state['dcount']), recursive=True)

    results = model(preds)
    # results.imgs
    results.render()
    os.mkdir("output/v{}".format(st.session_state['dcount']))
    for index,im in enumerate(results.imgs):
        
        img = Image.fromarray(im)
        img.save('output/v{}/{}'.format(st.session_state['dcount'], imgname[index]))

        st.image('output/v{}/{}'.format(st.session_state['dcount'], imgname[index]))

    st.button('Predict')

dcount = 0
if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False)
    model.classes = [0]
    main()
