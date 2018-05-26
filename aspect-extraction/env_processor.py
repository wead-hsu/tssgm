# -*- coding: utf-8 -*-
# ref: http://pyodps.readthedocs.io/zh_CN/latest/index.html

from odps import ODPS
#from preprocess import preprocessing
def make_env(params):
    return ODPSReader(params)

preprocessing = lambda x: x
class ODPSReader(object):
    def __init__(self, params):
        self.params = params
        self.odps_instance = ODPS(self.params['access_id'],
                                  self.params['access_key'],
                                  self.params['project_name'],
                                  self.params['end_point'])
        self.table = self.odps_instance.get_table(self.params['table'])


    def open(self):
        #with self.table.open_reader(partition=self.params['partition']) as reader:
        with self.table.open_reader() as reader:
            # read all records
            #start, end = 0, reader.count
            start, end = self.params["range_start"], self.params["range_end"]
            cur_reader = reader[start: end]
            for record in cur_reader:
                yield record, preprocessing(record)
    def count(self):
        #with self.table.open_reader(partition=self.params['partition']) as reader:
        with self.table.open_reader() as reader:
            return reader.count

    def __iter__(self):
        for sample in self.open():
            # 这里也可以做批量
            yield sample#, preprocessing(sample)
