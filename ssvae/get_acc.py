import sys

def get_acc(filename, by_loss):
    with open(filename, 'r') as f:
        res_valid_loss = 1e30
        res_valid_acc = 0
        res_test_acc = 0
        res_line_cnt = 0
        line_cnt = 0
        while True:
            line = f.readline()
            line_cnt += 1
            while line and 'VALIDATE:' not in line:
                line = f.readline()
                line_cnt += 1
            if not line:
                break
            valid_line = line
            test_line = f.readline()
            if not test_line:
                break
            line_cnt += 1
            assert 'TEST' in test_line
            #print(valid_line, test_line)
            valid_loss = float(valid_line.split('pred_l: ')[1].split('\t')[0])
            valid_acc = float(valid_line.split('acc_l: ')[1].split('\t')[0])
            test_acc = float(test_line.split('acc_l: ')[1].split('\t')[0])
            if by_loss:
                if valid_loss <= res_valid_loss:
                    res_line_cnt = line_cnt
                    res_valid_loss = valid_loss
                    res_valid_acc = valid_acc
                    res_test_acc = test_acc
            else:
                if valid_acc >= res_valid_acc:
                    res_line_cnt = line_cnt
                    res_valid_loss = valid_loss
                    res_valid_acc = valid_acc
                    res_test_acc = test_acc
        print('by_loss:', end='\t') if by_loss else print('by_acc :', end='\t')
        print('num_line: {}'.format(res_line_cnt), end='\t')
        print('valid_loss: {:.3f}'.format(res_valid_loss), end='\t')
        print('valid_acc: {:.3f}'.format(res_valid_acc), end='\t')
        print('test_acc: {:.3f}'.format(res_test_acc), end='\n')
    return res_valid_loss, res_valid_acc, res_test_acc, line_cnt

if __name__ == '__main__':
    for filename in  sys.argv[1::]:
        print(filename)
        get_acc(filename, True)
        get_acc(filename, False)
