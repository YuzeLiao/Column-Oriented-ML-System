#-------------------------------------------------------------------------------
# Copyright (c) 2016 by contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#-------------------------------------------------------------------------------

#!/usr/bin/env python
# -*- coding=utf-8 -*-

#-------------------------------------------------------------------------------
# Author : Chao Ma (mctt90@gmail.com) and Yuze Liao
# This file provides the command line option parser for f2m_main
#-------------------------------------------------------------------------------

from ConfigParser import ConfigParser
from util import CmdTool

class F2MOptionParser(object):
    """ Parse option from configure and generate a cmd_str
    """
    def __init__(self, config_file):
        self.config_file = config_file

    def Parse(self):
        config = ConfigParser()
        config.read(self.config_file);
        self.parameter = config.items("HyperParameter")

    def Run(self, main_file):
        cmd = main_file
        for key, value in self.parameter:
            if "file" not in key:
                value = value.lower()
            cmd += " --f2m_%s=%s"%(key, value)
        CmdTool.run_cmd_and_wait(CmdTool(), cmd)
