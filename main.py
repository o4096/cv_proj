import os, cv2
import numpy as np
import tkinter as tk
from tkinter     import filedialog
from tkinter     import *
from tkinter.ttk import *
from PIL import Image, ImageTk

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
			'Edge Detection',
			'Corner Detection',# Harris
			'SIFT keypoints',
			'Detection w/SIFT', #TODO: fix this so it properly detects all objects in the objects folder separately
			'Detection w/CNN',  #TODO: implement this, preferably with segmentation preprocessing somehow
			'Detection w/RCNN', #TODO: might scrap this one
		]

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
	
	def fltr_apply(self):
		idxs= self.fltr_list.curselection()
		if len(idxs)>0:
			filter= self.fltr_list.get(idxs[0])
			if filter not in self.filters:
				print(f'[INFO]: Filter {filter} not found.')
				return

			img= np.array(self.img_proc)

			print('[INFO]: Applying filter:', filter)
			if   filter=='Sharpen':        img= cv2.filter2D(img, -1, self.param_sharpness.get()*np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
			elif filter=='Gaussian Blur':  img= cv2.GaussianBlur(img, (5, 5), self.param_blur.get())
			elif filter=='Edge Detection': img= cv2.Canny(img, threshold1=self.param_canny_thresh1.get(), threshold2=self.param_canny_thresh2.get())
			elif filter=='Normalize':
				mini= np.min(img)
				diff= np.max(img)-mini
				if diff==0:
					print('[INFO]: Blank Image! Nothing to normalize.')
					return
				img= (((img-mini)/diff)*255).astype(np.uint8)
			elif filter=='Grayscale': img= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			elif filter=='SIFT keypoints':
				if len(img.shape)==3: gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				else:                 gray= img
				sift= cv2.xfeatures2d.SIFT_create()
				keypoints, descriptors= sift.detectAndCompute(gray, None)
				img= cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			elif filter=='Detection w/SIFT':
				for obj_image in os.listdir('objects'):
					obj_img= cv2.imread(os.path.join('objects', obj_image))
					if obj_img is None:
						continue
					if len(obj_img.shape)==3: obj_img= cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
					if len(img.shape)==3:     scn_img= cv2.cvtColor(img,     cv2.COLOR_BGR2GRAY)
					else:                     scn_img= img
					
					sift= cv2.SIFT_create()
					keypoints1, descriptors1= sift.detectAndCompute(scn_img, None)
					keypoints2, descriptors2= sift.detectAndCompute(obj_img, None)
					bf= cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
					matches= bf.match(descriptors1, descriptors2)
					matches= sorted(matches, key=lambda x: x.distance)

					# # Filter matches using Lowe's ratio test (optional)
					# good_matches= []
					# for m in matches:
					# 	if m.distance<0.75*np.mean([match.distance for match in matches]):
					# 		good_matches.append(m)

					# # Extract location of good matches
					# points_obj= np.zeros((len(matches), 2), dtype=np.float32)
					# points_scn= np.zeros((len(matches), 2), dtype=np.float32)

					# for i, match in enumerate(matches):
					# 	points_obj[i, :]= keypoints1[match.queryIdx].pt
					# 	points_scn[i, :]= keypoints2[match.trainIdx].pt

					# # # Find homography
					# H, mask= cv2.findHomography(points_obj, points_scn, cv2.RANSAC)

					# # # Get the corners of the object image
					# h, w= obj_img.shape[:2]
					# obj_corners= np.array([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]], dtype='float32').reshape(-1, 1, 2)

					# # # Transform corners to the scene image
					# scn_corners= cv2.perspectiveTransform(obj_corners, H)

					# # # Draw bounding box on the scene image
					# self.img2= cv2.polylines(self.img2, [np.int32(scn_corners)], isClosed=True, color=(0, 255, 0), thickness=3)

					if len(matches)>20:
						img= cv2.drawMatches(img, keypoints1, obj_img, keypoints2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
						img= cv2.putText(img, obj_image, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
			elif filter=='Harris Corner Detection':
				if len(img.shape)==3: gray= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				else:                 gray= img
				harris_response= cv2.cornerHarris(gray, self.param_harris_bsize.get(), self.param_harris_ksize.get(), self.param_harris_k.get())
				dilated_corners= cv2.dilate(harris_response, None)
				threshold= 0.01*harris_response.max()
				img[dilated_corners>threshold]= [0, 255, 0]# Mark corners in green
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
				tk.Scale(self.fltr_params, label='Sharpness', from_=0.1, to=2, resolution=0.1, orient='horizontal', variable=self.param_sharpness).pack()
			elif filter=='Gaussian Blur':
				tk.Scale(self.fltr_params, label='Sigma', from_=0.1, to=10, resolution=0.1, orient='horizontal', variable=self.param_blur).pack()
			elif filter=='Edge Detection':
				tk.Scale(self.fltr_params, label='Thresh1', from_=0, to=512, orient='horizontal', variable=self.param_canny_thresh1).pack()
				tk.Scale(self.fltr_params, label='Thresh2', from_=0, to=512, orient='horizontal', variable=self.param_canny_thresh2).pack()
			elif filter=='Normalize':        pass #no params
			elif filter=='Grayscale':        pass #no params
			elif filter=='SIFT keypoints':   pass #no params
			elif filter=='Detection w/SIFT': pass #no params (use objects folder for reference object images)
			elif filter=='Harris Corner Detection':
				tk.Scale(self.fltr_params, label='Block Size',  from_=2, to=10, orient='horizontal', variable=self.param_harris_bsize).pack()
				tk.Scale(self.fltr_params, label='Kernel Size', from_=1, to=9,   resolution=2,   orient='horizontal',  variable=self.param_harris_ksize).pack()
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
