#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 22:35:24 2023

@author: tungchentsai
"""

import os
import json
from . import widgets


with open(os.path.join(os.path.dirname(__file__), 'metadata.json')) as f:
    metadata = json.load(f)

vars().update(metadata)

