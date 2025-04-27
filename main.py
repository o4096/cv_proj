import cv2
import joblib
import numpy   as np
import tkinter as tk
from   tkinter import filedialog, messagebox
from   PIL     import Image, ImageTk
from   skimage.feature import hog

class Application:
	def __init__(self, root):
		self.root= root
		self.root.title('Computer Vision Application')
		self.root.geometry('800x600')
		self.img1= None
		self.img2= None
		self.model= joblib.load('svm_model.pkl')
		self.filters= [
			'Normalize', 'Grayscale', 
			'Sharpen', 'Gaussian Blur',
			'Edge Detection', 'Object Detection',
			# 'Resize', #introduces complications
			'SIFT', 'Harris Corner', 'OTSU',
			'Gaussian Pyramid', 'Laplacian Pyramid',
		]

		self.mb= tk.Menu(self.root) #menu bar construct
		self.mb_file= tk.Menu(self.mb, tearoff=0)
		self.mb_file.add_command(label='Open image', command=self.img_load)
		self.mb_file.add_command(label='Save image', command=self.img_save, state=tk.DISABLED)
		self.mb_file.add_separator()
		self.mb_file.add_command(label='Close image', command=self.img_clear, state=tk.DISABLED)
		self.mb_file.add_command(label='Exit',        command=self.root.quit)
		
		self.mb_about= tk.Menu(self.mb, tearoff=0)
		self.mb_about.add_command(label='About', command=lambda: messagebox.showinfo('Computer Vision Project', 'Helwan University 2025'))
		self.mb.add_cascade(label='File', menu=self.mb_file)
		self.mb.add_cascade(label='Help', menu=self.mb_about)
		self.root.config(menu=self.mb)

		self.f_l=     tk.Frame(self.root)
		self.f_l_img= tk.Label(self.f_l, text='No Image Loaded')

		self.f_l.pack(side='left', expand=1, fill='none', padx=0, pady=0)
		self.f_l.place(x=0, y=0)
		self.f_l_img.pack(side='top', expand=1, padx=0, pady=0)

		self.f_r= tk.Frame(self.root)

		self.f_r.pack(side=tk.RIGHT, fill='both', padx=2, pady=2)
		self.fltr_var=  tk.Variable(value=self.filters)
		self.fltr_list_title= tk.Label(self.f_r, text='Available Functions', justify='left')
		self.fltr_list_title.pack()
		self.fltr_list= tk.Listbox(self.f_r, selectmode='single', listvariable=self.fltr_var, state='disabled')
		self.fltr_list.pack(pady=5, padx=5, fill='both', expand=1)
		self.fltr_list.bind('<<ListboxSelect>>', self.fltr_select)

		self.rb= tk.Button(self.f_r, text='Revert Original', command=self.img_rvrtorigin, state='disabled')
		self.rb.pack()
		self.hb= tk.Button(self.f_r, text='Show Original',   command=self.img_showorigin, state='disabled')
		self.hb.pack()
		self.hb.bind('<ButtonPress-1>',   self.holdbutton_active)
		self.hb.bind('<ButtonRelease-1>', self.holdbutton_inactive)
		self.hb_active= False

		self.fltr_params_title= tk.Label(self.f_r, text='Parameters', justify='left')
		self.fltr_params_title.pack()
		self.fltr_params= tk.Frame(self.f_r)
		self.fltr_params.pack(pady=5, padx=5, fill='both', expand=1)

		self.param_sharpness= tk.DoubleVar(value=1.0)
		self.param_blur=      tk.DoubleVar(value=1.0)
		self.param_canny_thresh1= tk.IntVar(value=100)
		self.param_canny_thresh2= tk.IntVar(value=200)
		
		self.applyb= tk.Button(self.f_r, text='Apply', command=self.fltr_apply, state='disabled')
		self.applyb.pack(side='bottom')

	def _set_image(self, image):
		if image is None:
			return
		img_tk= ImageTk.PhotoImage(Image.fromarray(image))
		# img_pl.thumbnail((400, 400)) #stupid apis
		self.f_l_img.config(image=img_tk, text='', width=self.img1.shape[1], height=self.img1.shape[0])
		self.f_l_img.image= img_tk

	def clear_frame(self, frame:tk.Frame):
		for widget in frame.winfo_children():
			widget.destroy()

	def fltr_apply(self):
		idxs= self.fltr_list.curselection()
		if len(idxs)>0:
			filter= self.fltr_list.get(idxs[0])
			if filter not in self.filters:
				print(f'[INFO]: Filter {filter} not found.')
				return
			
			print('[INFO]: Applying filter:', filter)
			if   filter=='Sharpen':
				self.img2= cv2.filter2D(self.img2, -1, self.param_sharpness.get()*np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
			elif filter=='Gaussian Blur':
				self.img2= cv2.GaussianBlur(self.img2, (5, 5), self.param_blur.get())
			elif filter=='Edge Detection':
				self.img2= cv2.Canny(self.img2, threshold1=100, threshold2=200)
			elif filter=='Normalize':
				mini= np.min(self.img2)
				diff= np.max(self.img2)-mini
				if diff==0:
					print('[INFO]: Blank Image! Nothing to normalize.')
					return
				self.img2= (((self.img2-mini)/diff)*255).astype(np.uint8)
			elif filter=='Grayscale':
				self.img2= cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY)
			elif filter=='SIFT':
				if len(self.img2.shape)==3:
					gray= cv2.cvtColor(self.img2, cv2.COLOR_RGB2GRAY)
				else:
					gray= self.img2
				sift= cv2.xfeatures2d.SIFT_create()
				keypoints, descriptors= sift.detectAndCompute(gray, None)
				self.img2= cv2.drawKeypoints(self.img2, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
			elif filter=='Harris Corner': #TODO
				self.img2= cv2.cornerHarris(self.img2, blockSize=2, ksize=3, k=0.04)
			elif filter=='OTSU': #TODO
				self.img2= cv2.threshold(self.img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
			elif filter=='Object Detection': #TODO REVISE THIS!!!
				image= cv2.resize(self.img2, (64, 64))
				image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				image= cv2.GaussianBlur(image, (3, 3), 0)
				clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
				image= clahe.apply(image)
				_, thresh= cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
				kernel= np.ones((5, 5), np.uint8)
				morph= cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
				contours, _= cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				if contours:
					largest_contour= max(contours, key=cv2.contourArea)
					x, y, w, h= cv2.boundingRect(largest_contour)
					image= image[y:y+h, x:x+w]
					self.img2= cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)

				if len(image.shape)==3:
					image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				
				feat= hog(
					image, orientations=9,
					pixels_per_cell=(8, 8),
					cells_per_block=(2, 2),
					visualize=False, channel_axis=None
				)
				if feat.size!=489:
					if feat.size>489:
						feat= feat[:489]
					else:
						feat= np.pad(feat, (0, 489-feat.size), 'constant')

				# feat= pca.fit_transform(feat.reshape(1, -1))
				pred= self.model.predict(feat.reshape(1, -1))
				print(pred)
			else:
				return

			self.render()

	def fltr_select(self, _event):
		idxs= self.fltr_list.curselection()
		if len(idxs)>0:
			filter= self.fltr_list.get(idxs[0])
			if filter not in self.filters:
				print(f'[INFO]: Filter {filter} not found.')
				return

			self.clear_frame(self.fltr_params)
			self.applyb.config(state='normal')
			if   filter=='Sharpen':
				tk.Scale(self.fltr_params, from_=0.1, to=2, resolution=0.1, orient='horizontal', label='Sharpening Factor', variable=self.param_sharpness).pack()
			elif filter=='Gaussian Blur':
				tk.Scale(self.fltr_params, from_=0.1, to=10, resolution=0.1, orient='horizontal', label='Sigma', variable=self.param_blur).pack()
			elif filter=='Edge Detection':
				tk.Scale(self.fltr_params, from_=1, to=254, resolution=1, orient='horizontal', label='Threshold1', variable=self.param_canny_thresh1).pack()
				tk.Scale(self.fltr_params, from_=1, to=254, resolution=1, orient='horizontal', label='Threshold2', variable=self.param_canny_thresh2).pack()
			elif filter=='Normalize': pass #no params
			elif filter=='Grayscale': pass #no params
			elif filter=='SIFT':
				pass #TODO
			elif filter=='Harris Corner':
				pass #TODO
			elif filter=='OTSU':
				pass #TODO: segments the image
			elif filter=='Object Detection':
				pass #TODO: should draw bounding box around detected objects and show their class
			else:
				self.applyb.config(state='disabled')
				print(f'[INFO]: Filter {filter} not implemented.')
	
	def render(self):
		if self.img1 is None:
			self.f_l_img.config(image='', text='No Image Loaded')
		else:
			self._set_image(self.img2 if len(self.filters)>0 else self.img1)
		
	def holdbutton_active(self, event):
		self.hb_active= True
		self.hb.config(relief='sunken')
		self._set_image(self.img1)

	def holdbutton_inactive(self, event):
		self.hb_active= False
		self.hb.config(relief='raised')
		self.render()

	def img_rvrtorigin(self):
		self.img2= self.img1.copy()
		self.render()

	def img_showorigin(self):
		if self.hb_active:
			self.root.after(100, self.img_showorigin)
		else:
			self.render()

	def img_load(self):
		self.img_clear()
		fp= filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')])
		if fp:
			self.img1= cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
			self.img2= self.img1.copy()
			self.mb_file.entryconfig('Save image',  state='normal')
			self.mb_file.entryconfig('Close image', state='normal')
			self.fltr_list.config(state='normal')
			self.rb.config(state='normal')
			self.hb.config(state='normal')
			self.render()

	def img_save(self):
		fp= filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('BMP', '*.bmp')])
		if fp:
			cv2.imwrite(fp, cv2.cvtColor(self.img2, cv2.COLOR_RGB2BGR))
			print(f'[INFO]: Image saved to {fp}')
		else:
			print('[INFO]: Save image cancelled.')

	def img_clear(self):
		self.img1= None
		self.img2= None
		self.f_l_img.config(image='', text='No Image Loaded')
		self.f_l_img.image= None
		self.mb_file.entryconfig('Save image',  state='disabled')
		self.mb_file.entryconfig('Close image', state='disabled')
		self.fltr_list.config(state='disabled')
		self.rb.config(state='disabled')
		self.hb.config(state='disabled')
		self.applyb.config(state='disabled')
		print('[INFO]: Image Cleared.')

if __name__=='__main__':
	root= tk.Tk()
	app= Application(root)
	root.mainloop()
