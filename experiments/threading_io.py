#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 22:47:40 2024

@author: tungchentsai
"""

from pathlib import Path
from threading import Thread
from tempfile import TemporaryDirectory


def write_sequentially(n_files: int = 100):
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        for i in range(n_files):
            _write_large_file(folder / f'file_{i}.txt')


def write_from_threads(n_files: int = 100):
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        threads = []
        for i in range(n_files):
            threads.append(
                Thread(
                    target=_write_large_file,
                    args=(folder / f'file_{i}.txt',)
                )
            )
            threads[-1].start()
        
        # Wait until all finished
        for thread in threads:
            thread.join()


def _write_large_file(fpath):
    with open(fpath, 'w') as file:
        file.write('-' * 100_000_000)

