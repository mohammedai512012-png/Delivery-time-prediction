import torch
import pathlib
import os

def cmd_cleaner(work_by:{"cmd", "code"}, clean:bool=False):
    if work_by == "code" or work_by == None:
        if clean:
            os.system("cls" if os.name == "nt" else "clear")
    elif work_by == "cmd":
        g = input("Do you want to clear terminal before run the code (y/N): ")
        if g == "y":
            os.system("cls" if os.name == "nt" else "clear")
    else:
        raise ValueError("make sure that 'work_by' == 'cmd' or 'code'")

def save_model(model:torch.nn.Module,
               f: pathlib.Path):
    print(f"Saving model at {f}...")
    torch.save(obj=model, f=f)
    print("model saved successfully")

def load_model(model:torch.nn.Module,
               f:pathlib.Path):
    torch.load(model.load_state_dict(), f=f)