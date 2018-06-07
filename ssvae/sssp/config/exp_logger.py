import os
import time
import logging

def create_logger(logfn):
    """
    logging.basicConfig(filename=self.logfn, level=logging.INFO)
    self.logger = logging.getLogger()
    self.logger.addHandler(logging.StreamHandler())
    """

    #logging.basicConfig(level=logging.INFO)
    #logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(logfn)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    #consoleHandler = logging.StreamHandler()
    #consoleHandler.setFormatter(logFormatter)
    #rootLogger.addHandler(consoleHandler)
    return rootLogger

class ExpLogger(object):
    def __init__(self, lp='', ldir='../results', write_file=True):
        self.log_prefix = lp
        self.log_dir_path = ldir
        if write_file:
            self.logfn = os.path.join(self.log_dir_path, self.log_prefix + 'log.log')
            os.makedirs(self.log_dir_path)
            print('{} is created'.format(self.log_dir_path))
            self.logger = create_logger(self.logfn)
        else:
            self.logger = logging.getLogger()

    def write_args(self, args):
        self.message("-------- Parameter Info --------")
        sorted_args = sorted(args.items(), key=lambda x: x[0])
        for idx, item in enumerate(sorted_args):
            self.message("{}: {} = {}".format(str(idx), item[0], item[1]))
        self.message('--------------------')

    def message(self, str_line, write_file=True):
        str_stream = "{}: {}".format(self.log_prefix, str_line)
        if write_file:
            self.logger.info(str_stream)
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S") + str_stream)

    def write_variables(self, param_list):
        self.message("-------- Model Variables --------")
        for idx, param in enumerate(param_list):
            self.message(str(idx) + '. ' + str(param.name) + ': ' + str(param.get_shape()))
        self.message('--------------------')

    def file_copy(self, file_list):
        backup_path = os.path.join(self.log_dir_path, 'backup')
        #if not os.path.exists(backup_path):
            #os.makedirs(backup_path)
        for f in file_list:
            os.system('\\cp -rf ' + f + ' ' + backup_path + os.path.sep)

if __name__ == '__main__':
    explogger = ExpLogger('test', '/home/wdxu/tmp/test_explogger')
    explogger.message('1')
    explogger.message('1', True)
    print('2')
    #explogger.write_args({'a': 1, 'b': '2'})
