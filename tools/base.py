#!/usr/bin/python
# -*- coding: UTF-8 -*-

class Transpose(object):
    """
    This class is used for data transpose.
    """
    def __init__(self):
        self.data = []
        self.max_row_length = 0

    def line_process(self, line):
        """
        This method does some simple processing on each line of input data.
        """
        line = line.strip().split()
        if self.max_row_length < len(line):
            self.max_row_length = len(line)
        return line

    def line_extend(self, line):
        """
        This method extends a line's length to max_row_length.
        """
        extend_len = self.max_row_length - len(line)
        if extend_len > 0:
            line.extend(["0"] * extend_len)
        return line

    def read_data(self, input_filename):
        """
        This method reads data.
        """
        input_file = open(input_filename, "r")
        self.data = input_file.readlines()
        self.data = map(self.line_process, self.data)


    def data_transpose(self, data):
        """
        This method accomplishes the major work of data-transposition.
        """
        if data is not None:
            self.data = data
        self.data = map(self.line_extend, self.data)
        self.data = map(list, zip(*self.data))
        return self.data

    def data_print(self, output_filename):
        """
        This method prints data to a file.
        """
        output_file = open(output_filename, "w")
        for line in self.data:
            output_file.write(" ".join(line) + "\n")

class ColumnBasedTranspose(object):
    """
    This class is used to transfer data from
    row-based format to column-based format.
    """
    def __init__(self, batch_size, input_filename, output_filename):
        self.data = []
        self.max_length = 0
        self.instance_count = 0
        self.batch_size = batch_size
        self.global_max_length = 0
        self.transposer = Transpose()
        self.input_file = open(input_filename, "r")
        self.output_file = open(output_filename, "w")

    def read_data(self):
        """
        This method reads data.
        """
        self.data = []
        self.instance_count = 0
        while self.instance_count < self.batch_size:
            line = self.input_file.readline()
            if not line:
                break
            self.data.append(line)
            self.instance_count += 1

    def expand_line(self, line):
        """
        This method expands a line to a dense vector.
        """
        line = line.strip().split()
        last_idx = 0
        result = [line[0]]
        for item in line[1:]:
            [idx, val] = item.split(":")
            idx = int(idx)
            distance = idx - int(last_idx)
            if distance > 1:
                result.extend(["0"] * (distance - 1) + [val])
                last_idx = idx
            elif distance == 1:
                result.extend([val])
                last_idx = idx
            else:
                result[idx] = val
        if self.max_length < len(result):
            self.max_length = len(result)
        return result

    def expand_data(self):
        """
        This method expands data matrix.
        """
        while 1:
            self.read_data()
            if self.instance_count == 0:
                break
            self.max_length = 0
            self.data = map(self.expand_line, self.data)
            if self.global_max_length < self.max_length:
                self.global_max_length = self.max_length
            self.transposer.max_row_length = self.max_length
            self.data = self.transposer.data_transpose(self.data)
            self.data_print()
        self.output_file.write(str(self.global_max_length) + '\n')

    def data_print(self):
        """
        This method prints data to a file.
        """
        i = 0
        sub = 0
        while i < len(self.data):
            all_zero = 1
            for item in self.data[i]:
                if item != '0':
                    all_zero = 0
                    break
            if all_zero == 1:
                del self.data[i]
                sub += 1
            else:
                if i != 0:
                    self.data[i].insert(0, str(i + sub))
                i += 1
        if self.data:
            self.output_file.write(str(len(self.data)) + '\n')
            self.output_file.write(" ".join(self.data[0]) + '\n')
        for line in self.data[1:]:
            idx = 0
            self.output_file.write(line[0] + " ")
            for item in line[1:]:
                if item != "0":
                    self.output_file.write(str(idx) + ":" + item + " ")
                idx += 1
            self.output_file.write('\n')
