""" Many data samples are length-variable which makes random access
hard to implement. This class provides an interface to acces the 
data randomly by making a position tables for all samples.
"""

import sys
import os
import pickle as pkl
import logging
import threading

logger = logging.getLogger(__name__)

class RandomReader(object):
    def __init__(self, infilename,
            table_path=None):
        """ 
        Args:
            table_path: path of the table. If it is not set,
                tmp file will be created.
            delimiter: tag to split samples.
        """
        self._file = open(infilename, 'r')
        self._filename = infilename
        self._table_path = table_path
        if self._table_path is None:
            self._table_path = infilename + '.idx_cache'
        
        if os.path.exists(self._table_path):
            self._table = self._load_table(self._table_path)
        else:
            self._table = self._make_table(self._filename)
            logger.debug(self._table)
            self._save_table(self._table, self._table_path)
        self._num_samples = len(self._table)

    def __del__(self):
        self._file.close()

    @property
    def num_samples(self):
        return self._num_samples

    def _load_table(self, filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)

    def _save_table(self, table, filename):
        with open(filename, 'wb') as f:
            pkl.dump(table, f)

    def _make_table(self, infilename):
        table = []
        with open(infilename, 'rb') as f:
            while True:
                table.append(f.tell())
                if not f.readline():
                    table = table[:-1]
                    break
        return table

    def __getitem__(self, idx):
        # since multi-threading will acess _file object
        # simutanously, locking is required to ensure the
        # synchronization.
        lock = threading.Lock()
        lock.acquire()
        try:
            self._file.seek(self._table[idx])
            line = self._file.readline()
        finally:
            lock.release()
        return line

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rr = RandomReader('/home/wdxu/codes/textdream/data/skipthought/proc/train.idx.sub', 'table.pkl')
    print(rr.num_samples)
    print(rr[10000-1])
