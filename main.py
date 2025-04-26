import cv2
import numpy   as np
import tkinter as tk
from   tkinter import filedialog, messagebox
from   PIL     import Image, ImageTk

class ImageProcessor:
	# def __init__(self):
	# 	pass

	@staticmethod
	def normalize(image):
		# return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		mini= np.min(image)
		diff= np.max(image)-mini
		if diff==0:
			print('[INFO]: Image already normalized.')
			return image
		return (((image-mini)/diff)*255).astype(np.uint8)
	
	@staticmethod
	def resize(image, size=(100, 100), interpolation=cv2.INTER_AREA):
		return cv2.resize(image, size, interpolation=interpolation)

	@staticmethod
	def sharpen(image):
		return cv2.filter2D(image, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
	
	@staticmethod
	def gaussian_blur(image, kernel_size=(5, 5), sigma=1):
		return cv2.GaussianBlur(image, kernel_size, sigma)
		# return image.filter(ImageFilter.GaussianBlur(radius=sigma))
	
	@staticmethod
	def gaussian_pyramid(image, levels=3):
		gaussian_pyramid= [image]
		for i in range(levels-1):
			image= cv2.pyrDown(image)
			# image= image.filter(ImageFilter.GaussianBlur(radius=2**i))
			gaussian_pyramid.append(image)
		return gaussian_pyramid
	
	@staticmethod
	def laplacian_pyramid(image, levels=3):
		pass
		# laplacian_pyramid= []
		# gaussian_pyramid= ImageProcessor.gaussian_pyramid(image, levels)
		# for i in range(levels-1):
		# 	laplacian= gaussian_pyramid[i].filter(ImageFilter.GaussianBlur(radius=2**i))
		# 	laplacian= ImageChops.subtract(gaussian_pyramid[i], laplacian)
		# 	laplacian_pyramid.append(laplacian)
		# return laplacian_pyramid

	@staticmethod
	def edge_detection(image, threshold1=100, threshold2=200):
		return cv2.Canny(image, threshold1, threshold2)
	
	@staticmethod
	def object_detect(image):#TODO: Implement object detection
		return image

class Application:
	def __init__(self, root):
		self.root= root
		self.root.title('Computer Vision Application')
		self.root.geometry('800x600')
		self.img1= None
		self.img2= None

		'''#Polymorphism would complicate things, cause I want to tweak the params of each filter :')
		self.filters= {
			'Sharpen':          ImageProcessor.sharpen,
			'Gaussian Blur':    ImageProcessor.gaussian_blur,
			'Edge Detection':   ImageProcessor.edge_detection,
			'Normalize':        None,
			'Resize':           None,
			'Color Space':      None,
			'SIFT':             None,
			'Harris Corner':    None,
			'OTSU':             None,
			'Object Detection': ImageProcessor.object_detect,
		}'''
		self.filters= [
			'Sharpen', 'Gaussian Blur',
			'Edge Detection', 'Object Detection',
			'Normalize',
			# 'Resize',
			'Color Space', 
			'SIFT', 'Harris Corner', 'OTSU',
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

		self.f_l=      tk.Frame(self.root)
		self.f_l_img1= tk.Label(self.f_l, text='No Image Loaded')
		self.f_l_img2= tk.Label(self.f_l, text='')

		self.f_l.pack(side='left', expand=1, fill='none')
		self.f_l.place(x=0, y=0)
		self.f_l_img1.pack(side='top', expand=1, fill='both')
		self.f_l_img2.pack(side='top', expand=1, fill='both')

		self.f_r= tk.Frame(self.root)
		self.f_r.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2, pady=2)
		# self.fltr_var=  tk.Variable(value=list(self.filters.keys()))
		self.fltr_var=  tk.Variable(value=self.filters)
		self.fltr_list_title= tk.Label(self.f_r, text='Available Functions', justify=tk.LEFT)
		self.fltr_list_title.pack()
		self.fltr_list= tk.Listbox(self.f_r, selectmode=tk.MULTIPLE, listvariable=self.fltr_var)
		self.fltr_list.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
		self.fltr_list.bind('<<ListboxSelect>>', self.fltr_select)

		self.fltr_params_title= tk.Label(self.f_r, text='Parameters', justify=tk.LEFT)
		self.fltr_params_title.pack()
		self.fltr_params= tk.Frame(self.f_r)
		self.fltr_params.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

	def fltr_select(self, _event):
		idxs= self.fltr_list.curselection()
		if len(idxs)>0:
			fltrs= [self.fltr_list.get(i) for i in idxs]
			print('[INFO]: Selected filters:', fltrs)

			src= self.img1 if len(idxs)==1 else self.img2 #filter chaining

			for fltr in fltrs:
				if fltr in self.filters:
					print(f'[INFO]: Applying filter: {fltr}')
					if   fltr=='Sharpen':
						self.img2= ImageProcessor.sharpen(src)
					elif fltr=='Gaussian Blur':
						self.img2= ImageProcessor.gaussian_blur(src, kernel_size=(5, 5), sigma=1)
					elif fltr=='Edge Detection':
						self.img2= ImageProcessor.edge_detection(src, threshold1=100, threshold2=200)
					elif fltr=='Normalize':
						self.img2= ImageProcessor.normalize(src)
					# elif fltr=='Resize':
					# 	self.img2= ImageProcessor.resize(src)
					elif fltr=='Color Space':
						self.img2= cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
					elif fltr=='SIFT':
						self.img2= cv2.SIFT_create().detectAndCompute(src, None)[0]#TODO
					elif fltr=='Harris Corner':
						self.img2= cv2.cornerHarris(src, blockSize=2, ksize=3, k=0.04)#TODO
					elif fltr=='OTSU':
						self.img2= cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]#TODO
					elif fltr=='Object Detection':
						self.img2= ImageProcessor.object_detect(src)#TODO
					else:
						print(f'[INFO]: Filter {fltr} not implemented.')
				else:
					print(f'[INFO]: Filter {fltr} not found.')
		else:
			self.img2= self.img1
			print('[INFO]: No filters selected.')
		self.render()

	def render(self):
		if self.img1 is None:
			self.f_l_img1.config(image='', text='No Image Loaded')
		else:
			img_pl= Image.fromarray(self.img1)
			img_pl.thumbnail((400, 400)) #stupid apis
			img_tk= ImageTk.PhotoImage(img_pl)
			self.f_l_img1.config(image=img_tk, text='', width=self.img1.shape[1], height=self.img1.shape[0])
			self.f_l_img1.image= img_tk

		if self.img2 is None:
			self.f_l_img2.config(image='', text='No Image Loaded')
		else:
			img_pl= Image.fromarray(self.img2)
			img_pl.thumbnail((400, 400))
			img_tk= ImageTk.PhotoImage(img_pl)
			self.f_l_img2.config(image=img_tk, text='', width=self.img2.shape[1], height=self.img2.shape[0])
			self.f_l_img2.image= img_tk

	def img_load(self):
		self.img_clear()
		fp= filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')])
		if fp:
			self.img1= cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
			self.img2= self.img1.copy()
			self.mb_file.entryconfig('Save image',  state=tk.NORMAL)
			self.mb_file.entryconfig('Close image', state=tk.NORMAL)
			
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
		self.f_l_img1.config(image='', text='No Image Loaded')
		self.f_l_img2.config(image='', text='No Image Loaded')
		self.mb_file.entryconfig('Save image',  state=tk.DISABLED)
		self.mb_file.entryconfig('Close image', state=tk.DISABLED)
		print('[INFO]: Image Cleared.')

if __name__=='__main__':
	root= tk.Tk()
	app= Application(root)
	root.mainloop()
