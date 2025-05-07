import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter     import filedialog
from tkinter     import *
from tkinter.ttk import *
from PIL import Image, ImageTk
from keras._tf_keras.keras.models import load_model

IMG_SIZE= (224, 224, 3)
LABELS= {
        0: 'Unmasked',
        1: 'Masked',
}
model= load_model(os.path.join(os.path.dirname(__file__), 'model-018.keras'))	# get model path and load model
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # load Haar cascade classifier for face detection

class Application:
	def __init__(self, root):
		self.root= root
		self.root.title("Image Viewer")

		self.style= Style()
		self.style.theme_use('clam')

		self.img_orig=   None #Original image buffer
		self.img_proc=   None #Processed image buffer
		self.img_chosen= None #Reference to image chosen for drawing
		self.img_canvas= None #Canvas image item id
		self.zoom_scale= 1.0
		self.filters= [
			'Normalize', 'Grayscale',
			'Sharpen', 'Gaussian Blur',
			'Hist Equalization',
			'Hist Equalization',
			'Edge Detection',
			'Corner Detection', #Harris
			'Corner Detection', #Harris
			'SIFT keypoints',
			'OTSU Threshold',
			'Detection w/SIFT',
			'Detection w/CNN',
			'Kmeans Clustering',
			'OTSU Threshold',
			'Detection w/SIFT',
			'Detection w/CNN',
			'Kmeans Clustering',
		]
		self.model= load_model('chess.keras')

	# INIT UI ELEMENTS (order doesn't affect element position)
		self.canvas=            Canvas(root, bg='gray')
		self.ctrlpanel=         Frame(root)
		self.fltr_params=       Frame(self.ctrlpanel)
		self.fltr_list_title=   Label(self.ctrlpanel, text='Available Functions')
		self.fltr_params_title= Label(self.ctrlpanel, text='Parameters')
		self.fltr_var=          Variable(value=self.filters)
		self.fltr_list=         Listbox(self.ctrlpanel, selectmode=SINGLE, listvariable=self.fltr_var,      state=DISABLED)
		self.applyb=            Button(self.ctrlpanel, text='Apply',           command=self.fltr_apply,     state=DISABLED)
		self.rb=                Button(self.ctrlpanel, text='Revert Original', command=self.img_rvrtorigin, state=DISABLED)
		self.hb=                Button(self.ctrlpanel, text='Show Original',   command=None,                state=DISABLED)

	# DRAW UI ELEMENTS (ORDER MATTERS)
		self.canvas.pack(fill=BOTH, side=LEFT, expand=1, anchor=NW)
		self.ctrlpanel.pack(fill=BOTH, side=RIGHT, anchor=NE)
		self.fltr_list_title.pack(anchor=NW)
		self.fltr_list.pack(fill=BOTH)
		self.applyb.pack(side=BOTTOM)
		self.rb.pack(pady=1)
		self.hb.pack(pady=1)
		self.fltr_params_title.pack(anchor=NW)
		self.fltr_params.pack(fill=BOTH, side=LEFT, expand=1)

	# FILTER PARAMETERS
		self.param_sharpness=     DoubleVar(value=1.0)
		self.param_blur=          DoubleVar(value=1.0)
		self.param_canny_thresh1= IntVar(value=100)
		self.param_canny_thresh2= IntVar(value=200)
		self.param_harris_bsize=  IntVar(value=2)
		self.param_harris_ksize=  IntVar(value=3)
		self.param_harris_k=      DoubleVar(value=0.04)
		self.param_otsu_thresh=   DoubleVar(value=0)
		self.param_otsu_max=      DoubleVar(value=255)
		self.param_kmeans_k=      IntVar(value=2)
		self.param_otsu_thresh=   DoubleVar(value=0)
		self.param_otsu_max=      DoubleVar(value=255)
		self.param_kmeans_k=      IntVar(value=2)

		self.canvas.bind("<MouseWheel>", self.zoom)
		self.canvas.bind("<Button-1>",   self.img_load)
		self.fltr_list.bind('<<ListboxSelect>>', self.fltr_select)
		self.hb.bind('<ButtonPress-1>',   self.holdbutton_active)
		self.hb.bind('<ButtonRelease-1>', self.holdbutton_inactive)

	def holdbutton_active(self, event=None):
		self.img_chosen= self.img_orig
		self.img_display()

	def holdbutton_inactive(self, event=None):
		self.img_chosen= self.img_proc
		self.img_display()

	def img_rvrtorigin(self, event=None):
		self.img_proc= self.img_orig.copy()
		self.img_chosen= self.img_proc
		self.img_display()

	def fltr_grayscale(self, img):
		'''Returns grayscale version of colored images'''
		if len(img.shape)==3: gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		else:                 gray= img
		return gray

	def fltr_normalize(self, img):
		mini= np.min(img)
		diff= np.max(img)-mini
		if diff==0:
			print('[INFO]: Blank Image! Nothing to normalize.')
			return img
		return (((img-mini)/diff)*255).astype(np.uint8)

	def fltr_apply(self):
		idxs= self.fltr_list.curselection()
		if len(idxs)>0:
			filter= self.fltr_list.get(idxs[0])
			if filter not in self.filters:
				print(f'[INFO]: Filter {filter} not found.')
				return

			img= np.array(self.img_proc)

			print('[INFO]: Applying filter:', filter)
			if   filter=='Sharpen':           img= cv2.filter2D(img, -1, self.param_sharpness.get()*np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
			elif filter=='Gaussian Blur':     img= cv2.GaussianBlur(img, (5, 5), self.param_blur.get())
			elif filter=='Edge Detection':    img= cv2.Canny(img, threshold1=self.param_canny_thresh1.get(), threshold2=self.param_canny_thresh2.get())
			elif filter=='Normalize':         img= self.fltr_normalize(img)
			elif filter=='Grayscale':         img= self.fltr_grayscale(img)
			elif filter=='Hist Equalization':
				r, g, b= cv2.split(img)
				img= cv2.merge((cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)))
			if   filter=='Sharpen':           img= cv2.filter2D(img, -1, self.param_sharpness.get()*np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
			elif filter=='Gaussian Blur':     img= cv2.GaussianBlur(img, (5, 5), self.param_blur.get())
			elif filter=='Edge Detection':    img= cv2.Canny(img, threshold1=self.param_canny_thresh1.get(), threshold2=self.param_canny_thresh2.get())
			elif filter=='Normalize':         img= self.fltr_normalize(img)
			elif filter=='Grayscale':         img= self.fltr_grayscale(img)
			elif filter=='Hist Equalization':
				r, g, b= cv2.split(img)
				img= cv2.merge((cv2.equalizeHist(r), cv2.equalizeHist(g), cv2.equalizeHist(b)))
			elif filter=='SIFT keypoints':
				gray= self.fltr_grayscale(img)
				gray= self.fltr_grayscale(img)
				sift= cv2.xfeatures2d.SIFT_create()
				keypoints, descriptors= sift.detectAndCompute(gray, None)
				img= cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			elif filter=='OTSU Threshold':
				gray= self.fltr_grayscale(img)
				f, thresh= cv2.threshold(gray, self.param_otsu_thresh.get(), self.param_otsu_max.get(), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				img= thresh
			elif filter=='Corner Detection':
				gray= self.fltr_grayscale(img)
				harris_response= cv2.cornerHarris(gray, self.param_harris_bsize.get(), self.param_harris_ksize.get(), self.param_harris_k.get())
				dilated_corners= cv2.dilate(harris_response, None)
				threshold= 0.01*harris_response.max()
				if len(img.shape)==3: img[dilated_corners>threshold]= [0, 255, 0]# Mark corners in green
				else:                 img[dilated_corners>threshold]= 255
			elif filter=='OTSU Threshold':
				gray= self.fltr_grayscale(img)
				f, thresh= cv2.threshold(gray, self.param_otsu_thresh.get(), self.param_otsu_max.get(), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				img= thresh
			elif filter=='Corner Detection':
				gray= self.fltr_grayscale(img)
				harris_response= cv2.cornerHarris(gray, self.param_harris_bsize.get(), self.param_harris_ksize.get(), self.param_harris_k.get())
				dilated_corners= cv2.dilate(harris_response, None)
				threshold= 0.01*harris_response.max()
				if len(img.shape)==3: img[dilated_corners>threshold]= [0, 255, 0]# Mark corners in green
				else:                 img[dilated_corners>threshold]= 255
			elif filter=='Detection w/SIFT':
				scn_img= self.fltr_grayscale(img)
				scn_img= self.fltr_grayscale(img)
				for obj_image in os.listdir('objects'):
					obj_img= cv2.imread(os.path.join('objects', obj_image))
					if obj_img is None:
						continue
					obj_img= self.fltr_grayscale(obj_img)

					sift= cv2.SIFT_create()
					keypoints1, descriptors1= sift.detectAndCompute(scn_img, None)
					keypoints2, descriptors2= sift.detectAndCompute(obj_img, None)
					bf= cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
					matches= bf.match(descriptors1, descriptors2)
					matches= sorted(matches, key=lambda x: x.distance)

					# # # Filter matches using Lowe's ratio test (optional)
					# good_matches= []
					# for m in matches:
					# 	if m.distance<0.75*np.mean([match.distance for match in matches]):
					# 		good_matches.append(m)

					# # # Extract location of good matches
					# points_obj= np.zeros((len(matches), 2), dtype=np.float32)
					# points_scn= np.zeros((len(matches), 2), dtype=np.float32)

					for i, match in enumerate(matches):
						points_obj[i, :]= keypoints1[match.queryIdx].pt
						points_scn[i, :]= keypoints2[match.trainIdx].pt

					# H, mask= cv2.findHomography(points_obj, points_scn, cv2.RANSAC)

					# # # # Get the corners of the object image
					# h, w= obj_img.shape[:2]
					# obj_corners= np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype='float32').reshape(-1, 1, 2)

					# # # # Transform corners to the scene image
					# scn_corners= cv2.perspectiveTransform(obj_corners, H)

					# # # # Draw bounding box on the scene image
					# img= cv2.polylines(img, [np.int32(scn_corners)], isClosed=True, color=(0, 255, 0), thickness=3)
					if len(matches)>20:
						img= cv2.drawMatches(img, keypoints1, obj_img, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
						img= cv2.putText(img, obj_image, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			elif filter=='Detection w/CNN':
				faces= detector.detectMultiScale(img, 1.3, 5)
				for x, y, w, h in faces:
					crop= img[y:y+h, x:x+w]			        # crop the detected face
					crop= cv2.resize(crop, IMG_SIZE[:2])		# resize the cropped image to fit input size
					# crop= cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)	# convert color format to RGB
					crop= crop/255					# normalize pixels to range between 0 and 1
					crop= np.reshape(crop, [1, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]])
					preds= model.predict(crop)
					print(f'{preds}', end='\r')

					if preds[0]>=0.5:	# round the probability to get valid prediction
						label= 'Mask'
						color= (0, 255, 0) #GREEN
					else:
						label= 'No Mask'
						color= (255, 0, 0) #RED


				# gray= self.fltr_grayscale(img)
				# # blurred=     cv2.GaussianBlur(gray, (5, 5), 0) # Noise reduction
				# # equalized=   cv2.equalizeHist(blurred)
				# _, thresh=   cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				# # kernel=      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
				# # closed=      cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
				# contours, _= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				# for contour in contours:
				# 	# area= cv2.contourArea(contour)
				# 	# if area<1000 or area>1000000:
				# 	# 	continue
				# 	mask= np.zeros_like(gray)
				# 	cv2.drawContours(mask, [contour], -1, 255, -1)
				# 	x, y, w, h= cv2.boundingRect(contour)
				# 	if w<50 or h<50 or w>1000 or h>1000:
				# 		continue

				# 	crop= cv2.resize(img[y:y+h, x:x+w], IMG_SIZE)
				# 	crop= (crop/255).astype(np.float32)
				# 	preds= self.model.predict(np.array([crop]))
				# 	pred_idx= np.argmax(preds)
				# 	label= LABELS[pred_idx]
					img= cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
					img= cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=4)

				# mask= np.zeros_like(gray) #assume object has largest contour
				# if contours:
				# 	largest= max(contours, key=cv2.contourArea)
				# 	cv2.drawContours(mask, [largest], -1, 255, -1)
				# 	x, y, w, h= cv2.boundingRect(largest)

				# masked= cv2.bitwise_and(img, img, mask=mask)
				# img= self.fltr_normalize(masked)
				# img= cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0))
			elif filter=='Kmeans Clustering':
				data= np.float32(img.reshape((-1, 3)))
				criteria= (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
				_, labels, centers = cv2.kmeans(data, self.param_kmeans_k.get(), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
				centers= np.uint8(centers)
				segmented_data= centers[labels.flatten()]
				img= segmented_data.reshape(img.shape)
				# img= cv2.findChessboardCorners(img, (16, 16))
			else:
				return

			self.img_proc= Image.fromarray(img)
			self.img_chosen= self.img_proc
			self.img_display()

	def fltr_select(self, event=None):
		idxs= self.fltr_list.curselection()
		if len(idxs)>0:
			filter= self.fltr_list.get(idxs[0])
			if filter not in self.filters:
				print(f'[INFO]: Filter {filter} not found.')
				return

			for widget in self.fltr_params.winfo_children():
				widget.destroy()

			self.applyb.config(state=NORMAL)
			if   filter=='Sharpen':
				tk.Scale(self.fltr_params, label='Sharpness', from_=0.1, to=10, resolution=0.1, orient='horizontal', variable=self.param_sharpness).pack()
				tk.Scale(self.fltr_params, label='Sharpness', from_=0.1, to=10, resolution=0.1, orient='horizontal', variable=self.param_sharpness).pack()
			elif filter=='Gaussian Blur':
				tk.Scale(self.fltr_params, label='Sigma', from_=0.1, to=100, resolution=0.1, orient='horizontal', variable=self.param_blur).pack()
				tk.Scale(self.fltr_params, label='Sigma', from_=0.1, to=100, resolution=0.1, orient='horizontal', variable=self.param_blur).pack()
			elif filter=='Edge Detection':
				tk.Scale(self.fltr_params, label='Thresh1', from_=0, to=512, orient='horizontal', variable=self.param_canny_thresh1).pack()
				tk.Scale(self.fltr_params, label='Thresh2', from_=0, to=512, orient='horizontal', variable=self.param_canny_thresh2).pack()
			elif filter=='Normalize':        pass #no params
			elif filter=='Grayscale':        pass #no params
			elif filter=='SIFT keypoints':   pass #no params
			elif filter=='SIFT keypoints':   pass #no params
			elif filter=='SIFT keypoints':   pass #no params
			elif filter=='Detection w/SIFT': pass #no params (use objects folder for reference object images)
			elif filter=='OTSU Threshold':
				tk.Scale(self.fltr_params, label='Threshold', from_=0, to=255, orient='horizontal', variable=self.param_otsu_thresh).pack()
				tk.Scale(self.fltr_params, label='Max Value', from_=0, to=255, orient='horizontal', variable=self.param_otsu_max).pack()
			elif filter=='Detection w/CNN':  pass #no params
			elif filter=='Hist Equalization':pass #no params
			elif filter=='Kmeans Clustering':
				tk.Scale(self.fltr_params, label='K', from_=1, to=10, orient='horizontal', variable=self.param_kmeans_k).pack()
			elif filter=='Corner Detection':
			elif filter=='OTSU Threshold':
				tk.Scale(self.fltr_params, label='Threshold', from_=0, to=255, orient='horizontal', variable=self.param_otsu_thresh).pack()
				tk.Scale(self.fltr_params, label='Max Value', from_=0, to=255, orient='horizontal', variable=self.param_otsu_max).pack()
			elif filter=='Detection w/CNN':  pass #no params
			elif filter=='Hist Equalization':pass #no params
			elif filter=='Kmeans Clustering':
				tk.Scale(self.fltr_params, label='K', from_=1, to=10, orient='horizontal', variable=self.param_kmeans_k).pack()
			elif filter=='Corner Detection':
				tk.Scale(self.fltr_params, label='Block Size',  from_=2, to=10, orient='horizontal', variable=self.param_harris_bsize).pack()
				tk.Scale(self.fltr_params, label='Kernel Size', from_=1, to=9,   resolution=2,    orient='horizontal', variable=self.param_harris_ksize).pack()
				tk.Scale(self.fltr_params, label='Kernel Size', from_=1, to=9,   resolution=2,    orient='horizontal', variable=self.param_harris_ksize).pack()
				tk.Scale(self.fltr_params, label='k',           from_=0, to=0.1, resolution=0.01, orient='horizontal', variable=self.param_harris_k).pack()
			else:
				self.applyb.config(state=DISABLED)
				print(f'[INFO]: Filter {filter} not implemented.')

	def img_load(self, event=None):
		fp= filedialog.askopenfilename()
		if fp:
			self.img_orig= Image.open(fp)
			self.img_proc=   self.img_orig.copy()
			self.img_chosen= self.img_orig
			self.img_display()

			self.fltr_list.config(state=NORMAL)
			self.rb.config(state=NORMAL)
			self.hb.config(state=NORMAL)

	def img_display(self):
		w, h= self.img_chosen.size
		self.tk_image= ImageTk.PhotoImage(self.img_chosen.resize((int(w*self.zoom_scale), int(h*self.zoom_scale))))
		self.canvas.delete(ALL)
		self.img_canvas= self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)
		self.canvas.config(scrollregion=self.canvas.bbox(ALL))

	def zoom(self, event):
		if event.delta>0: self.zoom_scale*= 1.1
		else:             self.zoom_scale/= 1.1
		self.img_display()

if __name__=='__main__':
	root= Tk()
	app= Application(root)
	root.mainloop()
