"""
3D visualization of sponge/memory activations.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

def visualize_output(tensor, shape=(27, 27)):
    data = tensor.detach().cpu().numpy().reshape(shape)
    print("[#] Viewlayer Snapshot:")
    for row in data:
        print("".join("⬛" if v < 0.3 else "⬜" if v < 0.7 else "🟥" for v in row))

logger = logging.getLogger(__name__)

class ViewLayer:
    def __init__(self, memory):
        self.memory = memory

    def show_slice(self, key, axis=0, index=None):
        try:
            data = self.memory.load(key)
        except Exception as e:
            logger.warning(f"Cannot load slice '{key}': {e}")
            return
        if index is None:
            index = data.shape[axis] // 2
        slice_ = np.take(data, index, axis=axis)
        plt.imshow(slice_)
        plt.title(f"{key} slice axis={axis} idx={index}")
        plt.show()
