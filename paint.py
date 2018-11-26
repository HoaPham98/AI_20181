from tkinter import *
from tkinter import ttk, colorchooser, filedialog, messagebox
import PIL
from PIL import ImageGrab
import tensorflow as tf
import numpy as np
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.old_x = None
        self.old_y = None
        self.penwidth = 18
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)
        self.c.bind('<ButtonRelease-1>',self.reset)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def plotNumber(self, imageData, imageSize, prob):
        image = imageData.reshape(imageSize)
        print(prob)
        # root2 = Toplevel()
        # root2.title('number: {}'.format(prob))
        # fig2 = plt.Figure()
        # canvas2 = FigureCanvasTkAgg(fig2, master=root2)
        # canvas2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1.0)
        # ax = fig2.add_subplot(111)
        # ax.imshow(image)
        # canvas2.draw()
        # fig = plt.Figure()
        # self.c2 = FigureCanvasTkAgg(fig, master=self.canvases)
        # self.c2.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1.0)
        # # self.c2.grid(row=0, column=5, columnspan=5)
        # ax = fig.add_subplot(111)
        self.ax.imshow(image)
        self.fig.suptitle('number: {}'.format(prob))
        self.c2.draw()


    def reset(self,e):
        self.old_x = None
        self.old_y = None      

    def changeW(self,e):
        self.penwidth = e

    def save(self):
        x = self.c.winfo_rootx() + self.c.winfo_x() + 2
        y = self.c.winfo_rooty() + self.c.winfo_y() + 2
        x1 = x + self.c.winfo_width() - 4
        y1 = y + self.c.winfo_height() - 4

        filename = "paint.png"
        self.image = PIL.ImageGrab.grab().crop((x,y,x1,y1))
        self.image.save(filename)
        img = ndimage.imread(filename, mode='L')
        data = misc.imresize(img, (28,28), mode='L')
        # image = self.image.resize((28,28)).convert('L')
        # image.save(filename)
        misc.imsave(filename, data)
        data = data.reshape(1,28*28)
        self.recognize(data)
            
    def recognize(self, data):
        save_dir = './saveEmnist'
        loaded_Graph = tf.Graph()
        with tf.Session(graph=loaded_Graph) as sess:
            loader = tf.train.import_meta_graph(save_dir +'.meta')
            loader.restore(sess, save_dir)    
            # get tensors
            loaded_x = loaded_Graph.get_tensor_by_name('input:0')
            loaded_y = loaded_Graph.get_tensor_by_name('label:0')
            loaded_prob = loaded_Graph.get_tensor_by_name('probability:0')

            prob = sess.run(tf.argmax(loaded_prob,1), feed_dict = {loaded_x: data})
        
        self.plotNumber(data, (28,28), prob[0])

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
        
        self.canvases = Frame(self.master)
        self.c = Canvas(self.canvases,width=250,height=250,bg=self.color_bg,)
        self.c.grid(row=0, columnspan=5)

        self.fig = plt.Figure(figsize=(2.5,2.5))
        self.c2 = FigureCanvasTkAgg(self.fig, master=self.canvases)
        self.c2.get_tk_widget()
        self.c2.get_tk_widget().grid(row=0, column=5, columnspan=5)
        self.ax = self.fig.add_subplot(111)
        self.c2.draw()

        self.canvases.pack(fill=NONE)

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Nhom 02')
    root.resizable(False, False)
    root.mainloop()