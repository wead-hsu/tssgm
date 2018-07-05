- fclstm > sclstm > lstm
	- average of 5 runs = 0.7744: replicate1 decfclstm_unlabel10000_positionenc_bidirectionencFalse_positiondec_bidirectiondecFalse_trunctdis_sharefc
	- average of 5 runs = 0.7643: replicate2 declstm_unlabel10000_positionenc_bidirectionencFalse_positiondec_bidirectiondecFalse_trunctdis_sharefc
	- average of 5 runs = 0.7721: replicate3 decsclstm_unlabel10000_positionenc_bidirectionencFalse_positiondec_bidirectiondecFalse_trunctdis_sharefc

- depb ~= nodepb: softmax层的b是否依赖于y
	- average of 5 runs = 0.7744: replicate1 decfclstm_unlabel10000_positionenc_bidirectionencFalse_positiondec_bidirectiondecFalse_trunctdis_sharefc
	- average of 5 runs = 0.7714: replicate4 decfclstm_unlabel10000_positionenc_bidirectionencFalse_positiondec_bidirectiondecFalse_trunctdis_sharefc_depfcdepb

- fixemb > nofixemb

- filterdict > nofilterdict

- among all autoencoder fashions: None + bi + None + bi ~= None + uni + None + uni > others

- sharefc ~= nosharefc

- depb ~= nodepb
