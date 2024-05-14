import torch
import streamlit as st  # type: ignore
import os
import cv2
import time
from PIL import Image

# Assuming your helper and settings modules are in the same directory:
import helper
import settings


st.set_page_config(
    page_title="سیستەمی دەستنیشانکردنی تاسە",
    page_icon=":tada:",
    layout="wide",
)

cfg_model_path = "models/uploaded_YOLOv5m.pt"  # Replace with your actual path
model = None
confidence = 0.45  # Default value

def image_input(data_src):
    img_file = None
    if data_src == "نموونەی پێشوەختە":
        # Get all sample images
        img_path = glob.glob("data/sample_images/*")
        img_slider = st.slider("هەڵبژاردنی نموونەی پێشوەختە", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.sidebar.file_uploader("زیادکردنی وێنە", type=["png", "jpeg", "jpg"])
        if img_bytes:
            img_file = f"data/uploaded_data/upload.{img_bytes.name.split('.')[-1]}"
            Image.open(img_bytes).save(img_file)
        else:
            st.write(
                "<div dir='rtl'><p style='font-size: 30px; background-color: FireBrick; text-align: center;'>وێنەیەك زیادبکە</p></div>",
                unsafe_allow_html=True,
            )

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="وێنەی هەڵبژێردراو")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="تاسەی دەستنیشانکراو")


def video_input(data_src):
    vid_file = None
    try:
        if data_src == "نموونەی پێشوەختە":
            vid_file = "data/sample_videos/sample.mp4"
            cap = cv2.VideoCapture(vid_file)
            cap.release()

        else:
            vid_bytes = st.sidebar.file_uploader("زیادکردنی ڤیدۆ", type=["mp4", "mpv", "avi"])
            if vid_bytes:
                vid_file = f"data/uploaded_data/upload.{vid_bytes.name.split('.')[-1]}"
                with open(vid_file, "wb") as out:
                    out.write(vid_bytes.read())
            else:
                st.write(
                    "<div dir='rtl'><p style='font-size: 30px; background-color: FireBrick; text-align: center;'>ڤیدیۆیەك زیادبکە</p></div>",
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.sidebar.error(f":کێشە لە بارکردنی ڤیدیۆ هەیە {str(e)}")

    if vid_file:
        with open(vid_file, "rb") as video_file:
            video_bytes = video_file.read()

        if video_bytes:
            st.video(video_bytes)

        try:
            if vid_file:
                cap = cv2.VideoCapture(vid_file)
                custom_size = st.sidebar.checkbox("گۆڕینی قەبارەی چوارچێوە")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if custom_size: ####################################
                  width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
                  height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

                  fps = 0
                  st1, st2, st3 = st.columns(3)
                  with st1:
                      st.markdown("## بەرزی")
                      st1_text = st.markdown(f"{height}")
                  with st2:
                      st.markdown("## پانی")
                      st2_text = st.markdown(f"{width}")
                  with st3:
                      st.markdown("## FPS")
                      st3_text = st.markdown(f"{fps}")
  
                  st.markdown("---")
                  output = st.empty()
                  st1, st2 = st.columns(2)
  
                  with st1:
                      st.write(
                          "<div dir='rtl'><p style='font-size: 30px; background-color: green; text-align: center;'>ئەنجامی دۆزراوە</p></div>",
                          unsafe_allow_html=True,
                      )
                  with st2:
                      prev_time = 0
                      curr_time = 0
                      while True:
                          ret, frame = cap.read()
                          if not ret:
                              st.write(
                                  "<div dir='rtl'><p style='font-size: 30px; background-color: FireBrick; text-align: center;'>ڤیدیۆکە تەواو بوو تکایە نوێی بکەوە</p></div>",
                                  unsafe_allow_html=True,
                              )
                              break
                          frame = cv2.resize(frame, (width, height))
                          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                          output_img = infer_image(frame)
                          output.image(output_img)
                          curr_time = time.time()
                          fps = 1 / (curr_time - prev_time)
                          prev_time = curr_time
                          st1_text.markdown(f"**{height}**")
                          st2_text.markdown(f"**{width}**")
                          st3_text.markdown(f"**{fps:.2f}**")
  
                  cap.release()
          except Exception:
              st.write(
                  "<div dir='rtl'><p style='font-size: 30px; background-color: FireBrick; text-align: center;'>ڤیدیۆیەك زیادبکە</p></div>",
                  unsafe_allow_html=True,
              )
  
  
  def infer_image(img, size=None):
      if model is not None:  # Check if model is loaded successfully before using it
          model.conf = confidence
          result = model(img, size=size) if size else model(img)
          result.render()
          image = Image.fromarray(result.ims[0])
          return image
      else:
          print("Error: Model is not loaded. Please check for errors.")
          return None  # Handle the case where model is not available
  
  
  def load_model(cfg_model_path, device_option, pre_downloaded_weights_path=None):
      """Loads the YOLOv5 model from the specified path and device."""
      try:
          # Assuming `ultralytics.hub.load` is used:
          model = torch.hub.load(
              'ultralytics/yolov5', 'custom', source='local', path=pre_downloaded_weights_path or cfg_model_path
          )
          model.to(device_option)  # Move model to specified device
          return model
      except Exception as e:
          print(f"Error loading model: {e}")
          return None  # Handle the error gracefully or raise an exception
  
  
  def main():
      global model, confidence, cfg_model_path
  
      st.write("<p style='font-size: 50px; text-align: center; font-weight: 800;'>سیستەمی دەستنیشانکردنی تاسە</p>", unsafe_allow_html=True)
  
      st.sidebar.title("ڕێکخستنەکان")
  
      # Check if model file is available
      if not os.path.isfile(cfg_model_path):
          st.warning(".فایلی مۆدێل بەردەست نیە!!, تکایە زیادی بکە بۆ ناو فؤڵدەری مۆدێل", icon="⚠️")
      else:
          # Device options
          if torch.cuda.is_available():
              device_option = st.sidebar.radio("هەڵبژاردنی ئامێر", ['cpu', 'cuda'], disabled=False, index=0)
          else:
              device_option = st.sidebar.radio("هەڵبژاردنی ئامێر", ['cpu', 'cuda'], disabled=True, index=0)
  
          # Load model with error handling
          try:
              model = load_model(cfg_model_path, device_option)
              if model is None:
                  st.error("Error: Failed to load the model. Please check the logs for details.")
          except Exception as e:
              print(f"Error loading model: {e}")
              st.error(f"Error: An error occurred while loading the model. Check the console for details.")
  
          # Confidence slider
          confidence = st.sidebar.slider('ڕێژەی متمانە', min_value=0.1, max_value=1.0, value=.45)
  
          # Custom classes (optional)
          if st.sidebar.checkbox("پۆلە تایبەتەکان"):
              model_names = list(model.names.values())
              assigned_class = st.sidebar.multiselect("هەڵبژاردنی پؤلەکان", model_names, default=[model_names[0]])
              classes = [model_names.index(name) for name in assigned_class]
              model.classes = classes
          else:
              model.classes = list(model.names.keys())
  
          st.sidebar.markdown("---")
  
          # Input options
          input_option = st.sidebar.radio(":دیاریکردنی جۆری داخڵکردن", ['وێنە', 'ڤیدیؤ'])
  
          # Input src option
          data_src = st.sidebar.radio(":دیاریکردنی سەرچاوەی داخڵکردن", ['نموونەی پێشوەختە', 'داخڵکردنی نموونەی زیاتر'])
  
          # Process user input based on selected options
          if input_option == 'وێنە':
              image_input(data_src)
          elif input_option == 'ڤیدیؤ':
              video_input(data_src)
          # Add functionalities for webcam, RTSP stream, or YouTube video processing here (if needed)
          # elif input_option == 'وێبکام':
          #    helper.play_webcam(confidence, model)
          # elif input_option == 'rtsp':
          #    helper.play_rtsp_stream(confidence, model)
          # elif input_option == 'youtube':
          #    helper.play_youtube_video(confidence, model)
          else:
              st.error("!تکایە سەرچاوەی گونجاو دیاری بکە")
  
  if __name__ == "__main__":
      try:
          main()
      except SystemExit:
          pass

                  
                  
                    
