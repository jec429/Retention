import matplotlib
from feature_utils import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")


wwid = 1021037
df_all = pd.read_pickle('./data_files/job_full_title_cleaned_fixed_vectorized_similar.pkl')
df_br = pd.read_csv('data_files/Brazil_2018.csv', sep=',')
# df_br = pd.read_pickle('./data_files/job_full_title_cleaned_fixed_vectorized_similar.pkl')
x_merged = pd.read_pickle("./data_files/merged_Brazil_combined_x_numeric_new.pkl")

LARGE_FONT = ("Calibri", 16)


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

    def show_table(self, cont):
        frame = TablePage(self.container, self)
        self.frames[TablePage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()

    def show_plots(self, cont):
        frame = PlotPage(self.container, self)
        self.frames[PlotPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        from PIL import Image, ImageTk
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Retention", font="Calibri 28")
        label.pack(pady=10, padx=10)

        im = Image.open('./pics/JJ_Logo.jpg')
        basewidth = 300
        wpercent = (basewidth / float(im.size[0]))
        hsize = int((float(im.size[1]) * float(wpercent)))
        im = im.resize((basewidth, hsize), Image.ANTIALIAS)
        ph = ImageTk.PhotoImage(im)

        label_image = tk.Label(self, image=ph)
        label_image.image = ph
        label_image.place(relx=.5, rely=0.1, anchor="n")

        self.controller = controller
        self.treeview = None

        label_frame = tk.Label(self, text="WWID=", font=LARGE_FONT)
        label_frame.place(relx=.35, rely=0.255, anchor="n")

        self.entry = tk.Entry(self, width=10)
        self.entry.insert(1918, '1918')
        self.entry.place(relx=.5, rely=0.255, anchor="n")

        button1 = tk.Button(self, text="Get", command=self.jumpToFeature)
        button1.place(relx=.6, rely=0.255, anchor="n")

        button6 = tk.Button(self, text="Quit", command=self.quit)
        button6.configure(height=2, width=30)
        button6.place(relx=.47, rely=.8, anchor="c")

        label_title = tk.Label(self, text="Highest Risk", font=LARGE_FONT)
        label_title.place(relx=.5, rely=0.31, anchor="n")

        tv = ttk.Treeview(self)
        tv['columns'] = ('name', 'regret')
        tv.heading("#0", text='WWID')
        tv.column("#0", anchor="center", width=100)
        tv.heading('name', text='Name')
        tv.column('name', anchor='center', width=250)
        tv.heading('regret', text='Regrettable Loss')
        tv.column('regret', anchor='center', width=250)
        tv.place(relx=.47, rely=.55, anchor="c")

        import pickle
        fname = "parrot.pkl"
        with open(fname, "rb") as fin:
            list_lists2 = pickle.load(fin)
        wwids = list_lists2[0].to_list()
        prob_tf = list_lists2[2]
        print(wwids[:5])
        print(prob_tf[:5])
        prob_tf, wwids = zip(*sorted(zip(prob_tf, wwids), reverse=True))
        print(wwids[:5])
        print(prob_tf[:5])

        for iw in range(5):
            w = wwids[iw]
            df2 = df_all[df_all['WWID'] == w]
            if df2.size > 0:
                name = df2.iloc[0]['Name']
            else:
                df2 = df_br[df_br['WWID'] == w]
                if df2.size > 0:
                    name = df2.iloc[0]['Legal_Name']
                else:
                    name = 'N/A'

            tv.insert('', 'end', text=str(w), values=(name, 'No',))

    def jumpToFeature(self):
        global wwid
        print(int(self.entry.get()))
        wwid = int(self.entry.get())
        self.controller.show_table(TablePage)


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
                             command=lambda: controller.show_plots(PlotPage))
        button2.pack()

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
        # im = Image.open('./pics/'+str(random.randint(0, 22))+'.png')
        from pathlib import Path

        my_file = Path('./pics/' + str(wwid) + '.png')
        if my_file.is_file():
            # file exists
            im = Image.open('./pics/' + str(wwid) + '.png')
        else:
            im = Image.open('./pics/blank.png')
        im = Image.open('./pics/blank.png')
        basewidth = 300
        wpercent = (basewidth / float(im.size[0]))
        hsize = int((float(im.size[1]) * float(wpercent)))
        im = im.resize((basewidth, hsize), Image.ANTIALIAS)
        f.figimage(im, xo=50, yo=80)

        canvas = TkAgg.FigureCanvasTkAgg(f, master=view_nets)  # Moved Chart to view_nets Frame
        canvas.draw()
        # canvas.get_tk_widget().grid(column = 0, row = 0) I'll explain commenting this out below

        toolbar = TkAgg.NavigationToolbar2Tk(canvas, view_nets)
        toolbar.update()
        self.widget = canvas.get_tk_widget()
        self.widget.pack(fill='both', expand=False)

        # Adding Frame to bundle Treeview with Scrollbar (same idea as Plot+Navbar in same Frame)
        tableframe = tk.Frame(self)
        tableframe.pack(side='left', fill='x', expand=True)  # Packing against view_nets Frame

        # See Documentation for more info on Treeview
        table = ttk.Treeview(tableframe, show='headings')
        table.pack(side='left', fill='x')

        style = ttk.Style()
        style.configure("Treeview", highlightthickness=0, bd=0, font=('Calibri', 11))
        style.configure("Treeview.Heading", font=('Calibri', 13, 'bold'))

        table["columns"] = ("one", "two")
        table.column("one", width=200)
        table.column("two", width=300)
        table.heading("one", text="Feature")
        table.heading("two", text="Value")

        df2 = df_all[df_all['WWID'] == wwid]
        if df2.size > 0:
            name = df2.iloc[0]['Name']
            func = df2.iloc[0]['Function']
            sfunc = df2.iloc[0]['SubFunction']
        else:
            df2 = df_br[df_br['WWID'] == wwid]
            if df2.size > 0:
                name = df2.iloc[0]['Legal_Name']
                func = df2.iloc[0]['Job_Function__IA__Host_All_Other']
                sfunc = df2.iloc[0]['Job_Sub_Function__IA__Host_All_O']
            else:
                name = 'N/A'
                func = 'N/A'
                sfunc = 'N/A'

        import pickle
        fname = "websites.pkl"
        with open(fname, "rb") as fin:
            websites = pickle.load(fin)

        if wwid in websites:
            website = websites[wwid]
        else:
            website = 'N/A'

        table.insert('', 'end', values=("Name", name))
        table.insert('', 'end', values=("WWID", wwid))
        prob = calculate_probability(wwid)
        if 'High' in prob:
            table.insert('', 'end', values=("Resignation Probability", prob), tags=('highrow',))
        elif 'Medium' in prob:
            table.insert('', 'end', values=("Resignation Probability", prob), tags=('medrow',))
        elif 'Low' in prob:
            table.insert('', 'end', values=("Resignation Probability", prob), tags=('lowrow',))
        else:
            table.insert('', 'end', values=("Resignation Probability", prob))

        table.insert('', 'end', values=("Function", func))
        table.insert('', 'end', values=("SubFunction", sfunc))
        # table.insert('', 'end', values=("Website", website))
        table.tag_configure('highrow', background='lightcoral')
        table.tag_configure('medrow', background='peachpuff')
        table.tag_configure('lowrow', background='lightgreen')

        scroll = tk.Scrollbar(tableframe, command=table.yview)  # Adding Vertical Scrollbar
        scroll.pack(side='left', fill='y')
        table.configure(yscrollcommand=scroll.set)  # Attach Scrollbar

        f = 0


class PlotPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        self.widget = None
        if self.widget:
            self.widget.destroy()

        # names = [n[0] for n in data]
        global wwid
        h = histo_feature(wwid, x_merged)
        canvas = FigureCanvasTkAgg(h, self)
        canvas.draw()
        self.widget = canvas.get_tk_widget()

        self.widget.pack(fill=tk.BOTH, expand=True)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.place(relx=.4, rely=0.01, anchor="n")
        button_quit = ttk.Button(self, text="Quit", command=self.quit)
        button_quit.place(relx=.6, rely=0.01, anchor="n")

        button_table = ttk.Button(self, text="Table Page",
                                  command=lambda: controller.show_table(TablePage))
        button_table.place(relx=.5, rely=0.05, anchor="n")

        h = 0


app = FeaturesGUI()
app.mainloop()
