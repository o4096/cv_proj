import os, cv2
import numpy as np
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
		self.hb_active=  False
		self.zoom_scale= 1.0
		self.filters= [
			'Normalize', 'Grayscale', 
			'Sharpen', 'Gaussian Blur',
			'Edge Detection',
			'Detection w/SIFT',
			'SIFT keypoints',
			'Harris Corner Detector',
		]

	# INIT UI ELEMENTS (order doesn't affect element position)
		self.canvas=            Canvas(root, bg='gray')
		self.ctrlpanel=         Frame(self.root)
		self.fltr_params=       Frame(self.ctrlpanel)
		self.fltr_list_title=   Label(self.ctrlpanel, text='Available Functions')
		self.fltr_params_title= Label(self.ctrlpanel, text='Parameters')
		self.fltr_var=          Variable(value=self.filters)
		self.fltr_list=         Listbox(self.ctrlpanel, selectmode=SINGLE, listvariable=self.fltr_var,      state=DISABLED)
		self.applyb=            Button(self.ctrlpanel, text='Apply',           command=self.fltr_apply,     state=DISABLED)
		# self.rb=                Button(self.ctrlpanel, text='Revert Original', command=self.img_rvrtorigin, state=DISABLED)
		# self.hb=                Button(self.ctrlpanel, text='Show Original',   command=self.img_showorigin, state=DISABLED)

	# DRAW UI ELEMENTS (ORDER MATTERS)
		self.canvas.pack(fill=BOTH, expand=1)
		self.ctrlpanel.pack(fill=BOTH, side=RIGHT)
		self.fltr_list_title.pack(anchor=NW, expand=1)
		self.fltr_list.pack(fill=BOTH, expand=1)
		# self.rb.pack()
		# self.hb.pack()
		self.fltr_params_title.pack(side=LEFT, anchor=NW)
		self.fltr_params.pack(fill=BOTH, side=LEFT, expand=1)
		self.applyb.pack(side=BOTTOM)

	# FILTER PARAMETERS
		self.param_sharpness=     DoubleVar(value=1.0)
		self.param_blur=          DoubleVar(value=1.0)
		self.param_canny_thresh1= IntVar(value=100)
		self.param_canny_thresh2= IntVar(value=200)

		self.canvas.bind("<MouseWheel>", self.zoom)
		self.canvas.bind("<Button-1>",   self.img_load)
		self.fltr_list.bind('<<ListboxSelect>>', self.fltr_select)
		# self.hb.bind('<ButtonPress-1>',   self.holdbutton_active)
		# self.hb.bind('<ButtonRelease-1>', self.holdbutton_inactive)

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
			elif filter=='Harris Corner Detector':
				pass #TODO: detects all corner points in the image
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
				Scale(self.fltr_params, from_=0.1, to=2, orient='horizontal', variable=self.param_sharpness).pack()
			elif filter=='Gaussian Blur':
				Scale(self.fltr_params, from_=0.1, to=10, orient='horizontal', variable=self.param_blur).pack()
			elif filter=='Edge Detection':
				Scale(self.fltr_params, from_=1, to=254, orient='horizontal', variable=self.param_canny_thresh1).pack()
				Scale(self.fltr_params, from_=1, to=254, orient='horizontal', variable=self.param_canny_thresh2).pack()
			elif filter=='Normalize':        pass #no params
			elif filter=='Grayscale':        pass #no params
			elif filter=='SIFT keypoints':   pass #no params
			elif filter=='Detection w/SIFT': pass #TODO: pass a reference image of object to detect
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
			# self.rb.config(state=NORMAL)
			# self.hb.config(state=NORMAL)

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
