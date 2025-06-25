import tkinter as tk
from tkinter import filedialog ,Toplevel, IntVar, Checkbutton, Button
from tkinter import ttk
from tkinter import messagebox
from log import Log
from survey import Survey
import os
import pandas as pd
from tkinter import StringVar
#import matplotlib
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#from pycoustic import log  # Assuming log is an instance of a class with the required methods
test = 0

class Application(tk.Tk):
    
    def __init__(self):
        super().__init__()
        self.title("pycoustic Log Viewer")
        self.geometry("1200x800")
        import tkinter as tk

        # Create an instance of Survey
        self.survey = Survey()

        self.create_widgets()

    def create_widgets(self):

        self.grid_columnconfigure((0,1,2), weight=1)

        self.grid_columnconfigure((3,4,5), weight=2)

        self.log_label = ttk.Label(self, text="csv file name:")
        self.log_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        self.log_file = ttk.Label(self, text="Click Browse to select")
        self.log_file.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.browse_button1 = ttk.Button(self, text="Browse", command=self.browse_log)
        self.browse_button1.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        self.analysis_label = ttk.Label(self, text="Select Analysis Type:")
        self.analysis_label.grid(row=3, column=0, padx=5, pady=10, sticky="e")

        self.analysis_var = StringVar()
        self.analysis_combobox = ttk.Combobox(self, textvariable=self.analysis_var, values=["resi_summary", "modal_l90", "lmax_spectra", "Typical_leq_spectra"])
        self.analysis_combobox.set("resi_summary")
        self.analysis_combobox.grid(row=3, column=1, padx=5, pady=10, sticky="w")
        self.analysis_var.trace("w", self.on_analysischange)

        self.parameters_label = ttk.Label(self, text="Parameters")
        self.parameters_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")

        self.parameters_entry = ttk.Entry(self, width=20)
        self.parameters_entry.insert(0, 'None,None,10,2min')
        self.parameters_entry.grid(row=4, column=1, padx=5, pady=10, sticky="w")

        self.parameters_label = ttk.Label(self, text="(leq_cols,max_cols,lmax_n,lmax_t )")
        self.parameters_label.grid(row=4, column=2, padx=5, pady=5, sticky="w")

        self.execute_button = ttk.Button(self, text="Select Columns", command=self.Column_Selection_Modal)
        self.execute_button.grid(row=5, column=0, padx=5, pady=10, sticky="e")

        self.execute_button = ttk.Button(self, text="Execute", command=self.execute_code)
        self.execute_button.grid(row=5, column=2, padx=5, pady=10, sticky="w")

        self.tree = ttk.Treeview(self, show="headings")
        self.tree_scroll_y = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree_scroll_y.grid(row=6, column=6, sticky="ns")

        self.tree_scroll_x = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree_scroll_x.grid(row=7, column=0, columnspan=3, sticky="ew")

        self.tree.configure(yscrollcommand=self.tree_scroll_y.set, xscrollcommand=self.tree_scroll_x.set)

        self.tree.grid(row=6, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")
    """
      # Button to open the modal dialog
        open_dialog_button = ttk.Button(self, text="Show Time Series", command=self.open_modal_dialog)
        open_dialog_button.grid(row=5, column=2, padx=5, pady=10, sticky="w")


    # Commented out section that displays a graph
    def open_modal_dialog(self):
        dialog = Toplevel(self)
        dialog.title("Modal Dialog")

        # Make the dialog modal
        dialog.transient(self)
        dialog.grab_set()

        # Center the dialog on the screen
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        window_x = self.winfo_x()
        window_y = self.winfo_y()

        dialog_width = 900
        dialog_height = 600

        position_right = int(window_x + (window_width / 2) - (dialog_width / 2))
        position_down = int(window_y + (window_height / 2) - (dialog_height / 2))

        dialog.geometry(f"{dialog_width}x{dialog_height}+{position_right}+{position_down}")
        import matplotlib.pyplot as plt

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the data
        #print ("----",self.df.columns[1][0],self.df.columns[1][1] )        
        #print ("----",type(self.df.columns[1]))
        
        leq_a_column = [col for col in self.df.columns if isinstance(col, tuple) and col[0] == "Leq" and col[1] == "A"]
        if not leq_a_column:
            messagebox.showerror("Error", "No columns found with ('Leq', 'A') in the header.")
            return

        l90_a_columns = [col for col in self.df.columns if isinstance(col, tuple) and col[0] == "L90" and col[1] == "A"]
        if not l90_a_columns:
            messagebox.showerror("Error", "No columns found with ('L90', 'A') in the header.")
            return


        self.df["Date"] = pd.to_datetime(self.df.index)
        #print(self.df)

        ax.plot(self.df["Date"], self.df[leq_a_column[0]], label='LAeq')
        ax.plot(self.df["Date"], self.df[l90_a_column[0]], label='LA90')
        #ax.plot(self.df['Index'], self.df['L90 A'], label='la90')

        # Format the x-axis to show dates properly
        fig.autofmt_xdate()

        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Values')
        ax.set_title('Time Series Line Chart')
        ax.legend()

        # Create a canvas to display the plot in the Tkinter dialog
        canvas = FigureCanvasTkAgg(fig, master=dialog)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        #else:
        #    messagebox.showerror("Error", "DataFrame does not contain required columns: 'date', 'Laeq', 'la90'")

        # OK button to close the dialog
        ok_button = ttk.Button(dialog, text="OK", command=dialog.destroy)
        ok_button.pack(pady=20)

    """

    def on_analysischange(self, *args):
        analysis_type = self.analysis_combobox.get()
        if analysis_type == "resi_summary":
            self.parameters_entry.config(state='normal')
        else:
            self.parameters_entry.config(state='disabled')
        return
 
  
    def browse_log(self):
        self.logpath = os.getcwd  ()
        file_path = tk.filedialog.askopenfilename(initialdir=self.logpath, filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.logname = os.path.basename(file_path)
            self.logpath = os.path.dirname(file_path)
            self.log_file.config(text=self.logname)

            strPath = self.logpath + "\\" + self.logname
            self.log  = Log(path=strPath)
            self.survey.add_log(data=self.log, name="Position 1")

            self.df = self.log.get_data()

            # Clear the treeview
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Round the dataframe values to 1 decimal place, except for the column headers
            self.df = self.df.round(1)
 
            # Insert new data into the treeview
            self.tree["columns"] = ["Index"] + list(self.df.columns)
            self.tree.heading("Index", text="Index")
            self.tree.column("Index", width=150, stretch=True)

            i = 0 
            for col in self.df.columns:
                self.tree.heading(col, text=col)
                i=i+1
                if i < 12:
                    self.tree.column(col, width=75,stretch=True, anchor="center")
                else:
                    self.tree.column(col, width=0,stretch=False, anchor="center")
          
            for index, row in self.df.iterrows():
                self.tree.insert("", "end", values=[index] + list(row))
            return
 
    def execute_code(self):
        analysis_type = self.analysis_combobox.get()
        parameters = self.parameters_entry.get()

        try:
            df = pd.DataFrame()
            if analysis_type == "resi_summary":
                params = parameters.split(",")
                p = [None if x == "None" else x for x in params]
                df =self.survey.resi_summary(leq_cols=p[0], max_cols=p[1], lmax_n=int(params[2]), lmax_t=params[3])  
                print ("df id a ",type(df))
            elif analysis_type == "modal_l90":
                df = self.survey.modal()
            elif analysis_type == "lmax_spectra":
                df = self.survey.lmax_spectra()
            elif analysis_type == "Leq_spectra":
                df = self.survey.leq_spectra()
            else:
                messagebox.showerror("Error", "Please select an analysis type.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            return
        
        # Clear the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Set new column headers based on the dataframe
        #self.tree["columns"] = list(df.columns)

        # Insert new data into the treeview
        self.tree["columns"] = ["Index"] + list(df.columns)
        self.tree.heading("Index", text="Index")
        self.tree.column("Index", width=150, anchor="center", stretch=True)
        

        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=75, anchor="center", stretch=True)
   
        # Insert new data into the treeview
        for index, row in df.iterrows():
            self.tree.insert("", "end", values=[index] + list(row))

        # Copy the DataFrame to the clipboard
        df.to_clipboard(index=True)
        messagebox.showinfo("Success", "DataFrame copied to clipboard.")


    def Column_Selection_Modal(self):   
        # Create a modal dialog
        #print ("def Column_Selection_Modal(self):")

        dialog = Toplevel(self)
        dialog.title("Select Columns")

        # Make the dialog modal
        dialog.transient(self)
        dialog.grab_set()

        #self.window.update_idletasks()
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        window_x = self.winfo_x()
        window_y = self.winfo_y()

        dialog_width = 300
        dialog_height = 600

        position_right = int(window_x + (window_width / 2) - (dialog_width / 2))
        position_down = int(window_y + (window_height / 2) - (dialog_height / 2))

        dialog.geometry(f"{dialog_width}x{dialog_height}+{position_right}+{position_down}")

        # Configure grid rows and columns
        for i in range(12):  # Adjust the range as needed
            dialog.rowconfigure(i,weight =1,uniform = 'a')

        for i in range(3):  # Adjust the range as needed
            dialog.grid_columnconfigure(i,  weight =1,minsize = 20,uniform = 'a')

        # Get the column identifiers
        column_ids = self.tree["columns"]
        #print ("col ids",column_ids)

        # Get the column headers
        column_headers = [self.tree.heading(col)["text"] for col in column_ids]

        # Get the column widths
        column_widths = [self.tree.column(col)["width"] for col in column_ids]


        self.column_vars = {}
        for i, column in enumerate(column_headers):
            # Check if the column is visible in self.tree
            self.column_vars[column] = IntVar(master=dialog, value=1 if column_widths[i] > 0 else 0)
        self.cb = []

        # Create checkboxes for each column
        for i, column in enumerate(column_headers):
            #print(f"Column: {column}, Value: {self.column_vars[column].get()}")            
            self.cb.append( Checkbutton(dialog, text=column, variable=self.column_vars[column]))
            #self.cb[i].grid(row=i+1, column=(i-1)/10, sticky='w')
            self.cb[i].grid(row=i%10, column=i//10, sticky='w')

        for i, column in enumerate(column_headers):
            self.column_vars[column].set(1 if column_widths[i] > 0 else 0)

        # OK button to apply the selection
        ttk.Button(dialog, text="OK", command=lambda: self.apply_column_selection(dialog)).grid(row=13, column=1, ipady=10)
       # Cancel button
        def on_cancel():
            #print("Cancel clicked")
            dialog.destroy()

        cancel_button = ttk.Button(dialog, text="Cancel", command=on_cancel)
        cancel_button.grid(row=13,column = 2, sticky = "w",ipady = 10)   

        return     

    def apply_column_selection(self, dialog):

        #print("def apply_column_selection")

        # Get the selected columns
        selected_columns = [column for column, var in self.column_vars.items() if var.get() == 1]

        # Set the column widths based on the selection
        for column in self.tree["columns"]:
            column_text = self.tree.heading(column)["text"]
            if column_text in selected_columns:
                self.tree.column(column, width=250,stretch=True, anchor="center")  # Set to a default width
            else:
                self.tree.column(column, width=0,stretch=False, anchor="center")  # Hide the column

        # Display the selected columns in the listbox
        #for item in self.tree:
        #    display_text = " | ".join(str(item[column]) for column in selected_columns)
        #    self.listbox.insert(END, display_text)
        # Close the dialog
        dialog.destroy()

if __name__ == "__main__":
    app = Application()
    app.mainloop()