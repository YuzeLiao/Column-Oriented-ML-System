#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
from base import ColumnBasedTranspose

TRANS = ColumnBasedTranspose(int(sys.argv[3]), sys.argv[1], sys.argv[2])
TRANS.expand_data()
TRANS.data_print()
