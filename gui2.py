from __future__ import print_function
from Tkinter import *
from Tkinter import PhotoImage
import os
from PIL import Image, ImageTk

class Gui:
	def __init__(self,master):
		frame=Frame(master, width=600, height=600,bg='grey')
		frame.grid(row=0,column=1)
		#frame.config();
		#C = Canvas(root, bg="blue", height=600, width=600)
		#filename = ImageTk.PhotoImage(Image.open("/Users/ashwin/Desktop/network-security.png"))
		#background_label = Label(master, image=filename)
		#background_label.image = filename
		#background_label.place(x=0, y=0, relwidth=1, relheight=1)
		#background_label.pack()
		#C.pack()

		self.button_ids=Button(frame, text='Intrusion Detection System', width=25,height=3,command=self.ids,bg="red", fg='white')
		self.button_ids.pack(padx=250,pady=20)
		self.button_anomaly=Button(frame, text='Anomaly Detection System', width=25, height=3,command=self.anomaly, bg='blue', fg='white')
		self.button_anomaly.pack(padx=250,pady=120)
		self.button_quit=Button(frame, text='Exit', command=quit,height=3, width=25,bg='green')
		self.button_quit.pack(padx=250,pady=25)
	def ids(self):
		os.system('python decision-tree-classifier.py dataset/all.csv dataset/jan17.csv');
	def anomaly(self):
		os.system('python anomaly.py')

canvas_width = 300
canvas_height =300


root=Tk()
#s = Style();
#s.configure('Myframe',background='red');
root.geometry("700x500")
root.title("Intelligent Network Security")



gui=Gui(root)
root.mainloop()