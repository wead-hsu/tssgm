# -*- coding: utf-8 -*-

## ref: http://gitlab.alipay-inc.com/junlong.qjl/reinforcement-learning/blob/master/policy_monitor.py
from odps import ODPS
from odps.models import Schema, Column, Partition

def make_writer(params):
    return ODPSWriter(params)

class ODPSWriter(object):
    def __init__(self, params):
        self.params = params
        self.odps_instance = ODPS(self.params['access_id'],
                         self.params['access_key'],
                         self.params['project_name'],
                         self.params['end_point'])
        self.odps_table = self.create_table()


    def create_table(self):

        table_name = self.params['table']
        table_project = self.params['project_name']
        #table_partition = self.params['partition']
        table_exist = self.odps_instance.exist_table(table_name, table_project)

        if not table_exist:
            raise Exception('Table does not exist!')

        table = self.odps_instance.get_table(table_name)
        #table.delete_partition(table_partition, if_exists=True)  # delete partition if exists

        #table.create_partition(table_partition, if_not_exists=True)  # create partition if not exist
        print('Create partition successfully!')
        return table


    def write_record(self, record, block_id):
        with self.odps_table.open_writer(blocks=[block_id], reopen=True) as writer:
            writer.write(block_id, record)
