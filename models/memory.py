import torch
from collections import deque

class ShortMemory:

    def __init__(self, config):
        self.memory_size = config['memory_config']['short_size']
        self.queue = deque(maxlen=self.memory_size)
            
    def pop(self, element):
        if len(self.queue) == self.memory_size:
            self.queue.popleft()
        self.queue.append(element)
    
    def init(self, element):
        if len(self.queue) == self.memory_size:
            for _ in range(self.memory_size):
                self.queue.popleft()
        for _ in range(self.memory_size):
            self.queue.append(element)
        
    def read_memory(self):
        return torch.cat(list(self.queue), dim=1)
