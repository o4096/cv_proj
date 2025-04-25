import cv2
import numpy   as np
import tkinter as tk
from   tkinter import filedialog, messagebox
from   PIL     import Image, ImageTk, ImageFilter

class Application:
	def __init__(self, root):
		self.root= root
		self.root.title('Image Processor')
		self.img= None
		self.img_proc= None
		self.filters= {
			'Sharpen':          ImageFilter.SHARPEN,
			'Gaussian Blur':    ImageFilter.GaussianBlur,
			'Edge Detection':   ImageFilter.FIND_EDGES,
			'Object Detection': None,
			'Normalize':        None,
			'Resize':           None,
		}

		menubar= tk.Menu(self.root) #menu bar construct
		mb_file= tk.Menu(menubar, tearoff=0)
		mb_file.add_command(label='Load image', command=self.img_load)
		mb_file.add_command(label='Save image', command=self.img_save)
		mb_file.add_separator()
		mb_file.add_command(label='Exit',       command=self.root.quit)

		mb_about= tk.Menu(menubar, tearoff=0)
		mb_about.add_command(label='About', command=lambda: messagebox.showinfo('Image Processor v1.0', 'Helwan University 2025'))
		menubar.add_cascade(label='File', menu=mb_file)
		menubar.add_cascade(label='Help', menu=mb_about)
		self.root.config(menu=menubar)

		# self.image_label= tk.Label(self.root, text='No Image Loaded', width=100, height=50)
		# self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		self.frame_img= tk.Frame(self.root)
		self.frame_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)
		self.frame_img_proc= tk.Frame(self.root)
		self.frame_img_proc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

		self.ctrl_frame= tk.Frame(self.root)
		self.ctrl_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=2, pady=2)
		
		self.filter_var=  tk.Variable(value=list(self.filters.keys()))
		self.filter_list_title= tk.Label(self.ctrl_frame, text='Available Functions')
		self.filter_list_title.pack()
		self.filter_list= tk.Listbox(self.ctrl_frame, selectmode=tk.MULTIPLE, listvariable=self.filter_var)
		self.filter_list.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
		self.filter_list.bind('<<ListboxSelect>>', self.filter_select)

		self.filter_params_title= tk.Label(self.ctrl_frame, text='Parameters')
		self.filter_params_title.pack()
		self.filter_params= tk.Frame(self.ctrl_frame)
		self.filter_params.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

	def filter_select(self, event):
		idxs= self.filter_list.curselection()
		if len(idxs)>0:
			print('[INFO]: Last Selected filter:', self.filter_list.get(idxs[-1]))
		else:
			print('[INFO]: No filters selected.')
		#TODO
		self.img_display(self.img_proc)

	def render(self):
		pass

	def img_load(self):
		fp= filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg;*.bmp')])
		if fp:
			self.img= Image.open(fp)
			self.img_display(self.img)

	def img_save(self):
		fp= filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg'), ('BMP', '*.bmp')])
		if fp:
			self.img_proc.save(fp)
			print(f'[INFO]: Image saved to {fp}')
		else:
			print('[INFO]: Save image cancelled.')

	def img_display(self, img):
		img.thumbnail((400, 400))
		img_tk= ImageTk.PhotoImage(img)
		self.image_label.config(image=img_tk, text='')
		self.image_label.image = img_tk

	def img_filter(self):
		if not self.img:
			print('[INFO]: No image loaded!')
			return

		selected_filter= self.filter_combobox.get()
		if selected_filter=='Blur':
			self.img_proc = self.img.filter(ImageFilter.BLUR)
		elif selected_filter=='Edge Detection':
			self.img_proc = self.img_filter_edge_detection()
		elif selected_filter=='Object Detection':
			self.img_proc = self.img_filter_object_detect()
		else:
			print('[INFO]: Please select a valid filter!')
			return

		self.img_display(self.img_proc)

	def img_filter_edge_detection(self):
		img=   cv2.cvtColor(np.array(self.img), cv2.COLOR_RGB2BGR)
		edges= cv2.Canny(img, 100, 200)
		return Image.fromarray(edges)

	def img_filter_object_detect(self):#TODO: Implement object detection
		print('[INFO]: Object detection not implemented yet.')
		return self.img

if __name__=='__main__':
	root= tk.Tk()
	app= Application(root)
	root.mainloop()
