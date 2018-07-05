import pickle as pkl
v = pkl.load(open('vocab_sent.pkl', 'rb'))
cnt = 0
for item in v:
    cnt += 1
    print(item)
    if cnt == 10: break
