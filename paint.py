from tkinter import *
from tkinter import ttk, colorchooser, filedialog, messagebox
import PIL
from PIL import ImageGrab
import tensorflow as tf
import numpy as np


class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 14
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)
        self.c.bind('<ButtonRelease-1>',self.reset)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):
        self.old_x = None
        self.old_y = None      

    def changeW(self,e):
        self.penwidth = e

    def save(self):
        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()

        filename = "paint.png"
        self.image = PIL.ImageGrab.grab().crop((x,y,x1,y1)).convert('1')
        image = self.image.resize((28,28))
        image.save(filename)
        data = np.asarray(image)
        data = data.reshape(1,28*28)
        self.recognize(data)
            
    def recognize(self, data):
        save_dir = './save'
        loaded_Graph = tf.Graph()
        with tf.Session(graph=loaded_Graph) as sess:
            loader = tf.train.import_meta_graph(save_dir +'.meta')
            loader.restore(sess, save_dir)    
            # get tensors
            loaded_x = loaded_Graph.get_tensor_by_name('input:0')
            loaded_y = loaded_Graph.get_tensor_by_name('label:0')
            loaded_prob = loaded_Graph.get_tensor_by_name('probability:0')

            prob = sess.run(tf.argmax(loaded_prob,1), feed_dict = {loaded_x: data})
        
        messagebox.showinfo("","Predict: Number {0}".format(prob))

    def reg(self):
        self.save()
        

    def clear(self):
        self.c.delete(ALL)

    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)

        self.eraser_button = Button(self.controls, text='eraser', command=self.clear)
        self.eraser_button.grid(row=0, column=0)

        self.save_button = Button(self.controls, text="recognize", comman=self.reg)
        self.save_button.grid(row=0,column=1)
        self.controls.pack()
        
        self.c = Canvas(self.master,width=250,height=250,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)
        
        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Nhom 02')
    root.resizable(False, False)
    root.mainloop()
