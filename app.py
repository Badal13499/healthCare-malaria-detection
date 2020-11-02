import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import cv2

def main():
	st.title("Malaria Detection Using Cell Images App")
	st.subheader("By Badal Kumawat")

	model = load_model("model_vgg19.h5")
	
	st.header("Choose a Cell Image to Check the Infection")
	st.set_option('deprecation.showfileUploaderEncoding', False)
	img_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
	if img_file is not None:
		our_image = Image.open(img_file)
		st.image(our_image, width=224, height=224)
	
		img = cv2.resize(np.float32(our_image),(224,224))
		y = image.img_to_array(img)
		y = y/255
		y = np.expand_dims(y, axis=0)
		img_data = preprocess_input(y)
		if st.button("Predict"):
			st.success("Model is Working....")
			y_preds = model.predict(img_data)
			y_pred = np.argmax(y_preds, axis=1)
			a = sum(y_pred)
			if a==1:
  				st.subheader("Uninfected")
			else:
  				st.subheader("Infected")  

if __name__ == "__main__":
	main()

