import numpy as np
from tensorboardX import SummaryWriter


def progress(_progress):
    bar_length = 5  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(_progress, int):
        _progress = float(_progress)
    if not isinstance(_progress, float):
        _progress = 0
        status = "error: progress var must be float\r\n"
    if _progress < 0:
        _progress = 0
        status = "Halt...\r\n"
    if _progress >= 1:
        _progress = 1
        status = ""
    block = int(round(bar_length * _progress))
    text = "\r\t[%s]\t%.2f%% %s" % (
            "#" * block + " " * (bar_length-block), _progress * 100, status)

    return text

