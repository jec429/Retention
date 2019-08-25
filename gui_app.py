import matplotlib
from feature_utils import *

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
import numpy as np
import matplotlib.pyplot as plt


wwid = 1918
df = pd.read_pickle('./data_files/job_full_title_cleaned_fixed_vectorized_similar.pkl')
x_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_new.pkl")

LARGE_FONT = ("Verdana", 12)


class FeaturesGUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Retention App")

        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        # container.pack()

        self.frames = {}

        for F in (StartPage, PageOne, TablePage, PlotPage):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def show_Table(self, cont):
        frame = TablePage(self.container, self)
        self.frames[TablePage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        self.controller = controller

        labelF = tk.Label(self, text="WWID=", font=LARGE_FONT)
        labelF.place(relx=.35, rely=0.255, anchor="n")
        self.entry = tk.Entry(self, width=10)
        self.entry.insert(1918, '1918')
        self.entry.place(relx=.5, rely=0.25, anchor="n")

        button1 = tk.Button(self, text="Get", command=self.jumpToFeature)
        button1.place(relx=.6, rely=0.255, anchor="n")

        # button2 = tk.Button(self, text="Table Page", command=lambda: controller.show_Table(TablePage))
        # button2.configure(height=2, width=20)
        # button2.place(relx=.35, rely=0.4, anchor="c")
        #
        # button3 = tk.Button(self, text="Plots Page", command=lambda: controller.show_frame(PlotPage))
        # button3.configure(height=2, width=20)
        # button3.place(relx=.65, rely=0.4, anchor="c")
        #
        # button4 = tk.Button(self, text="Boost Page", command=lambda: controller.show_frame(BoostPage))
        # button4.configure(height=2, width=40)
        # button4.place(relx=.5, rely=.5, anchor="c")
        #
        # button5 = tk.Button(self, text="Reset status", command=self.reset_status)
        # button5.configure(height=2, width=40)
        # button5.place(relx=.5, rely=.6, anchor="c")

        button6 = tk.Button(self, text="Quit",
                            command=self.quit)
        button6.configure(height=2, width=40)
        button6.place(relx=.5, rely=.7, anchor="c")

    def reset_status(self):
        global fstatus
        fstatus = len(fstatus) * [1]
        print(fstatus)

    def jumpToFeature(self):
        global wwid
        print(int(self.entry.get()))
        wwid = int(self.entry.get())
        self.controller.show_Table(TablePage)


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                             command=lambda: controller.show_frame(TablePage))
        button2.pack()


class TablePage(tk.Frame):

    def __init__(self, parent, controller):
        import matplotlib.backends.backend_tkagg as TkAgg
        import matplotlib.image as mpimg
        from PIL import Image
        import random

        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="TABLE", font=LARGE_FONT)
        label.pack(pady=5, padx=5)

        global wwid

        self.widget = None
        if self.widget:
            self.widget.destroy()

        self.controller = controller
        self.parent = parent
        self.treeview = None

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Plots",
                             command=lambda: controller.show_frame(PlotPage))
        button2.pack()

        # button3 = ttk.Button(self, text="Plots Page",
        #                      command=lambda: controller.show_frame(PlotPage))
        # button3.pack()
        #
        # buttonN = ttk.Button(self, text="Next",
        #                      command=self.nextTable)
        # buttonN.pack()
        #
        # buttonP = ttk.Button(self, text="Previous",
        #                      command=self.previousTable)
        # buttonP.pack()
        #
        # buttonPrint = ttk.Button(self, text="Print Table",
        #                          command=self.printTable)
        # buttonPrint.pack()
        #
        # buttonU = tk.Button(self, text="Update status", command=lambda: update_status(fnameHDF, fstatus))
        # buttonU.pack()

        buttonQ = ttk.Button(self, text="Quit",
                             command=self.quit)
        buttonQ.pack(pady=5, padx=5)

        view_nets = tk.Frame(self)
        view_nets.pack(side='left', fill='both', expand=False)  # Packing Left in order to place
        # another Frame next to it

        # Widgets
        f = plt.Figure(figsize=(4, 5), dpi=100)
        F = f.add_subplot(111)
        F.axis('off')
        print('wwid=', wwid)
        # im = Image.open('./pics/'+str(random.randint(0, 23))+'.png')
        im = Image.open('./pics/1.png')
        basewidth = 300
        wpercent = (basewidth / float(im.size[0]))
        hsize = int((float(im.size[1]) * float(wpercent)))
        im = im.resize((basewidth, hsize), Image.ANTIALIAS)
        height = im.size[1]
        f.figimage(im, xo=50, yo=10)

        canvas = TkAgg.FigureCanvasTkAgg(f, master=view_nets)  # Moved Chart to view_nets Frame
        canvas.draw()
        # canvas.get_tk_widget().grid(column = 0, row = 0) I'll explain commenting this out below

        toolbar = TkAgg.NavigationToolbar2Tk(canvas, view_nets)
        toolbar.update()
        self.widget = canvas.get_tk_widget()
        self.widget.pack(fill='both', expand=False)

        # Adding Frame to bundle Treeview with Scrollbar (same idea as Plot+Navbar in same Frame)
        tableframe = tk.Frame(self)
        tableframe.pack(side='left', fill='x', expand=True)  ## Packing against view_nets Frame

        # See Documentation for more info on Treeview
        table = ttk.Treeview(tableframe, show='headings')
        table.pack(side='left', fill='x')

        style = ttk.Style()
        style.configure("Treeview", highlightthickness=0, bd=0, font=('Calibri', 11))
        style.configure("Treeview.Heading", font=('Calibri', 13, 'bold'))

        table["columns"] = ("one", "two")
        table.column("one", width=200)
        table.column("two", width=200)
        table.heading("one", text="Feature")
        table.heading("two", text="Value")

        df2 = df[df['WWID'] == wwid]
        name = df2.iloc[0]['Name']

        table.insert('', 'end', values=("Name", name))
        table.insert('', 'end', values=("WWID", wwid))
        self.calculate_probability(wwid)
        table.insert('', 'end', values=("Resignation Probability", self.calculate_probability(wwid)))

        scroll = tk.Scrollbar(tableframe, command=table.yview)  ## Adding Vertical Scrollbar
        scroll.pack(side='left', fill='y')
        table.configure(yscrollcommand=scroll.set)  ## Attach Scrollbar

        f = 0

        #self.CreateUI()
        #self.LoadTable(tindex)

    def CreateUI(self):
        if self.treeview:
            self.treeview.destroy()
        tv = ttk.Treeview(self)
        tv['columns'] = ('index', 'mean', 'std', 'status')
        tv.heading("#0", text='Feature', anchor='w')
        tv.column("#0", anchor="w")
        tv.heading('index', text='Index')
        tv.column('index', anchor='center', width=50)
        tv.heading('mean', text='Mean')
        tv.column('mean', anchor='center', width=100)
        tv.heading('std', text='STD')
        tv.column('std', anchor='center', width=100)
        tv.heading('status', text='Status')
        tv.column('status', anchor='center', width=50)
        tv.pack(pady=6, padx=6)
        self.treeview = tv

    def LoadTable(self, tindex):
        global fstatus
        ldata = readDataHDFBlock(fnameHDF, tindex)
        # print(ldata[17])
        for i, d in enumerate(ldata):
            if fstatus[tindex * 20 + i] == -1 or fstatus[tindex * 20 + i] == 1:
                fstatus[tindex * 20 + i] = 0 if np.std(d[1:]) == 0 else 1
            self.treeview.insert('', 'end', text=d[0], values=(
                tindex * 20 + i,
                '%.3f' % np.mean(d[1:]),
                '%.3f' % np.std(d[1:]),
                u'\u2705' if fstatus[tindex * 20 + i] == 1 else u'\u274C')
                                 )

    def nextTable(self):
        global tindex
        tindex += 1
        self.DeleteUI()
        self.CreateUI()
        self.LoadTable(tindex)

    def previousTable(self):
        global tindex
        if tindex == 0: return
        tindex -= 1
        self.DeleteUI()
        self.CreateUI()
        self.LoadTable(tindex)

    def DeleteUI(self):
        self.treeview.delete()

    def calculate_probability(self, wwid):
        import pickle
        from sklearn.preprocessing import StandardScaler

        x = x_merged[(x_merged['Report_Year'] == 2018) & (x_merged['Working_Country'] == 37)]
        x = x.drop(['Report_Year', 'Working_Country'], axis=1)
        x = x.drop(['Status'], axis=1)
        x = x.reset_index(drop=True)
        i = x.index[x['WWID'] == wwid].tolist()
        print(i)
        if len(i) == 0:
            return 'N/A'
        x = x.drop(['WWID'], axis=1)
        x = np.array(x.values)
        x2 = StandardScaler().fit_transform(x)
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        prob = loaded_model.predict_proba(x2[i])
        return '%.2f' % prob[0][1]


class PlotPage(tk.Frame):

    def __init__(self, parent, controller):
        global wwid
        print('WWID=', wwid)
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        self.widget = None
        if self.widget:
            self.widget.destroy()

        # names = [n[0] for n in data]
        h = self.histo_feature()
        canvas = FigureCanvasTkAgg(h, self)
        canvas.draw()
        self.widget = canvas.get_tk_widget()

        self.widget.pack(fill=tk.BOTH, expand=True)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.place(relx=.4, rely=0.01, anchor="n")
        buttonQ = ttk.Button(self, text="Quit", command=self.quit)
        buttonQ.place(relx=.6, rely=0.01, anchor="n")

        buttonT = ttk.Button(self, text="Table Page",
                             command=lambda: controller.show_Table(TablePage))
        buttonT.place(relx=.5, rely=0.5, anchor="n")

        h = 0

    def histo_feature(self):
        # x = x_merged[(x_merged['Report_Year'] == 2018) & (x_merged['Working_Country'] == 37)]
        # x = x.drop(['Report_Year', 'Working_Country'], axis=1)
        # x = x.drop(['Status'], axis=1)
        # i = x.index[x['WWID'] == wwid].tolist()
        # if len(i) == 0:
        #     return 'N/A'
        # x = x.drop(['WWID'], axis=1)
        # x = np.array(x.values)
        #
        # means = x.mean(0)
        # stds = x.std(0)
        # sel = x[i]
        # print(sel)

        x = np.linspace(0, 2 * np.pi, 400)
        y = np.sin(x ** 2)

        fig, axs = plt.subplots(2, 2)
        fig
        axs[0, 0].plot(x, y)
        axs[0, 0].set_title('Axis [0, 0]')
        axs[0, 1].plot(x, y, 'tab:orange')
        axs[0, 1].set_title('Axis [0, 1]')
        axs[1, 0].plot(x, -y, 'tab:green')
        axs[1, 0].set_title('Axis [1, 0]')
        axs[1, 1].plot(x, -y, 'tab:red')
        axs[1, 1].set_title('Axis [1, 1]')

        return fig


app = FeaturesGUI()
app.mainloop()
