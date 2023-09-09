import tkinter as tk
from tkinter import ttk
import torch
import torch.nn as nn
import torch.optim as optim


class App:
    def __init__(self):
        self.points = []
        self.loss: float = 0.0

        self.root: tk.Tk = tk.Tk()
        self.root.title('Linear Regression')
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.main_frame: ttk.Frame = ttk.Frame(self.root,
                                               padding='3 3 3 3')
        self.main_frame.grid()
        self.main_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                                          weight=1)
        self.main_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5,
                                              6, 7, 8, 9, 10),
                                             weight=1)

        self.canva: tk.Canvas = tk.Canvas(self.main_frame,
                                          bg='white',
                                          width=400,
                                          height=400,
                                          borderwidth=3,
                                          relief='raised')
        self.canva.grid(column=0,
                        row=0,
                        columnspan=10,
                        rowspan=10)
        self.canva.bind('<Button-1>',
                        self.add_point)

        self.button_start: tk.Button = tk.Button(self.main_frame,
                                                 text='start',
                                                 command=self.start)
        self.button_start.grid(column=10,
                               row=9)

        self.button_reset: tk.Button = tk.Button(self.main_frame,
                                                 text='reset',
                                                 command=self.reset)
        self.button_reset.grid(column=10,
                               row=7)

        self.loss_label: tk.Label = tk.Label(self.main_frame,
                                             text='loss')
        self.loss_label.grid(column=10,
                             row=4)

        self.loss_str: tk.StringVar = tk.StringVar()
        self.loss_str.set(str(self.loss))

        self.loss_str_label: tk.Label = tk.Label(self.main_frame,
                                                 textvariable=self.loss_str)
        self.loss_str_label.grid(column=10,
                                 row=5)

        for child in self.main_frame.winfo_children():
            child.grid(sticky='NWES',
                       padx=(0, 3),
                       pady=(0, 3))

        self.root.mainloop()

    def start(self):
        data: torch.tensor = torch.tensor(self.points).float()
        x: torch.tensor = data[:, 0]
        y: torch.tensor = data[:, 1]
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=1)
        model = nn.Linear(1, 1)
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.000001)
        for epoch in range(100):
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss = loss.item()
            self.loss_str.set(str(int(self.loss)))
        self.draw_line(model.weight.item(), model.bias.item())

    def add_point(self, event: tk.Event):
        x: int = event.x
        y: int = event.y
        self.points.append((x, y))
        self.canva.create_line((x - 3, y - 3, x + 3, y + 3),
                               width=2,
                               fill='red')
        self.canva.create_line((x - 3, y + 3, x + 3, y - 3),
                               width=2,
                               fill='red')

    def draw_line(self, a: float, b: float):
        self.canva.create_line((0, b, 400, 400 * a + b),
                               width=2,
                               fill='black')

    def reset(self):
        self.points = []
        self.canva.delete('all')
        self.loss_str.set(str(0.0))
