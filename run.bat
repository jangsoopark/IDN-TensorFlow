: x2 pre
: python main.py --model_name=idn_pre_x2 --is_train=True --scale=2 --pretrain=False --epochs=100 --data_path=data/train_data/idn_train_x2.h5
: python main.py --model_name=idn_pre_x2 --is_train=False --scale=2

: x2 fine tune
: python main.py --model_name=idn_x2 --is_train=True --scale=2 --pretrain=True --pretrained_model_name=idn_pre_x2 --learning_rate_decay=True --decay_step=250 --epochs=600 --data_path=data/train_data/idn_fine_tuning_x2.h5 
: python main.py --model_name=idn_x2 --is_train=False --scale=2

: x3 pre
: python main.py --model_name=idn_pre_x3 --is_train=True --scale=3 --pretrain=False --epochs=100 --data_path=data/train_data/idn_train_x3.h5
: python main.py --model_name=idn_pre_x3 --is_train=False --scale=3

: x3 fine tune
: python main.py --model_name=idn_x3 --is_train=True --scale=3 --pretrain=True --pretrained_model_name=idn_pre_x3 --learning_rate_decay=True --decay_step=250 --epochs=600 --data_path=data/train_data/idn_fine_tuning_x3.h5 
: python main.py --model_name=idn_x3 --is_train=False --scale=3

: x4 pre
: python main.py --model_name=idn_pre_x4 --is_train=True --scale=4 --pretrain=False --epochs=100 --data_path=data/train_data/idn_train_x4.h5
: python main.py --model_name=idn_pre_x4 --is_train=False --scale=4

: x4 fine tune
: python main.py --model_name=idn_x4 --is_train=True --scale=4 --pretrain=True --pretrained_model_name=idn_pre_x4 --learning_rate_decay=True --decay_step=250 --epochs=600 --data_path=data/train_data/idn_fine_tuning_x4.h5 
: python main.py --model_name=idn_x4 --is_train=False --scale=4
