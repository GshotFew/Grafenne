## Code


Use following command for training and testing on single snapshot data

`python train.py --data=CiteSeer --missing_rate=0.99 --result_file=tmp.txt --gpu=0 --verbose=1 --num_epochs=300 --num_layers=2 --bs_train_nbd=-1 --bs_test_nbd=-1'
`

Use below command for training and testing on streaming graph in continual setup

`python continual.py --data=CiteSeer --gpu=0`


The above commands are CiteSeer, similar can be for other datasets such as Cora