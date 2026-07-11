import os
import numpy as np
import torch
from torch.utils.data import IterableDataset

class BinaryArrayDataset(IterableDataset):
    def __init__(self, filepath: str, record_size: int, block_size: int, batch_size: int):
        """
        An optimized, multi-worker-safe streaming dataset that reads contiguous blocks
        from a binary file via memory mapping, shuffles them in RAM, and yields pre-formed batches.
        
        Args:
            filepath (str): Path to the compiled raw binary dataset (.bin).
            record_size (int): Size of a single position record in bytes (default: 72).
            block_size (int): Number of positions to load contiguously into RAM at once.
            batch_size (int): Size of the mini-batches to slice out of the loaded block.
        """
        super().__init__()
        self.filepath = filepath
        self.record_size = record_size
        self.batch_size = batch_size
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Binary dataset file not found at: {filepath}")
            
        if block_size % batch_size != 0:
            raise ValueError(f"block_size ({block_size}) must be perfectly divisible by batch_size ({batch_size})")
        self.block_size = block_size  

        file_bytes = os.path.getsize(filepath)
        self.total_positions = file_bytes // self.record_size
        self.num_blocks = self.total_positions // self.block_size
        
        if self.num_blocks == 0:
            raise ValueError("The binary file size is smaller than a single block_size allocation.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process training execution (num_workers=0)
            start_idx = 0
            end_idx = self.num_blocks
            step = 1
        else:
            # Multi-process data loading execution (num_workers > 0)
            start_idx = worker_info.id
            end_idx = self.num_blocks
            step = worker_info.num_workers
            
        # specifically assigned block indices for this worker
        worker_blocks = np.arange(start_idx, end_idx, step)
        np.random.shuffle(worker_blocks)
        
        mmap_data = np.memmap(self.filepath, dtype=np.uint8, mode='r')
        
        for block_idx in worker_blocks:
            start_pos = block_idx * self.block_size
            start_byte = start_pos * self.record_size
            end_byte = (start_pos + self.block_size) * self.record_size
            
            block_bytes = np.copy(mmap_data[start_byte:end_byte])
            block = block_bytes.reshape(self.block_size, self.record_size)
            
            # shuffle block locally within system memory
            np.random.shuffle(block)
            
            for i in range(0, self.block_size, self.batch_size):
                batch_slice = block[i : i + self.batch_size]
                
                batch = torch.from_numpy(batch_slice.copy()).long()
                
                yield batch