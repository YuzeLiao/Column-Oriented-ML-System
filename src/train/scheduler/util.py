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
# Author : Chao Ma (mctt90@gmail.com)
# This file provides the utilities.
#-------------------------------------------------------------------------------

import inspect
import logging
import os
import sys
import subprocess
import traceback
import urllib

class CmdTool(object):
    """ Command line tool, used to
    1. run command in local Bash
    2. run ssh command on remote machine
    3. copy files to remote machines via scp
    """
    def __init__(self, ssh_port=22):
        self.ssh_port = ssh_port

    def display(self, mesg):
        """ display message and caller
        """
        caller = '<%s.%s>' %(self.__class__.__name__, inspect.stack()[1][3])
        mesg = '%s: %s' %(caller, mesg)
        print(mesg)
        sys.stdout.flush()

    def set_ssh_port(self, ssh_port):
        self.ssh_port = ssh_port

    def wait_cmd(self, process, cmd_str):
        """ wait for the process running cmd
        """
        retcode = process.wait()
        if retcode != 0:
            mesg = 'Fail with retcode(%s): %s' %(retcode, cmd_str)
            raise RuntimeError(mesg)

    def run_cmd(self, cmd_str):
        """ Run local command
        """
        cmd_lines = [line for line in cmd_str.splitlines() if len(line) > 0]
        cmd_str = ' \\\n'.join(cmd_lines)
        os.environ['PATH'] = '/usr/local/bin:/bin:/usr/bin:/sbin/'
        process = subprocess.Popen(cmd_str, shell=True, env=os.environ)
        mesg = 'run command PPID=%s PID=%s CMD=%s' %(os.getpid(), process.pid, cmd_str)
        logging.debug(mesg)
        return process

    def run_cmd_and_wait(self, cmd_str):
        process = self.run_cmd(cmd_str)
        self.wait_cmd(process, cmd_str)

    def run_ssh_cmd(self, machine, remote_cmd):
        """ Run command via SSH in remote machines
        """
        ssh = 'ssh -q -p %s' % self.ssh_port
        ssh_cmd = '%s %s \'%s\'' %(ssh, machine, remote_cmd)
        return self.run_cmd(ssh_cmd)

    def dispatch_file(self, file_list, dir_dict):
        """ Copy source files to remote machines
        """
        files = ' '.join(file_list)
        job_list = []
        for machine, tmp_dir in dir_dict.items():
            cmd_mkdir = 'mkdir -p %s' % tmp_dir
            job_list.append([self.run_ssh_cmd(machine, cmd_mkdir),
                            cmd_mkdir])
        for process, cmd_scp in job_list:
            self.wait_cmd(process, cmd_scp)

        job_list = []
        for machine, tmp_dir in dir_dict.items():
            cmd_scp = ('scp -q -P %s %s %s:%s/ >/dev/null'
                      % (self.ssh_port, files, machine, tmp_dir))
            job_list.append([self.ssh_port, files, machine, tmp_dir])
        for process, cmd_scp in job_list:
            self.wait_cmd(process, cmd_scp)

