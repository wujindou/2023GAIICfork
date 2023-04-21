class Config(dict):
    def version_config(self, version):
        batch = 64
        val_batch = 16
        hp = {1: {'n_epoch': 401, 'batch': batch, 'valid_batch': val_batch, 'n_layer':6},
              2: {'n_epoch': 401, 'batch': batch, 'valid_batch': val_batch, 'n_layer':6}
             }
        self['n_epoch'] = hp[version].get('n_epoch', 401)
        self['pre_n_epoch'] = hp[version].get('n_epoch', 401)
        self['n_layer'] = hp[version].get('n_layer', 6)
        self['batch'] = hp[version].get('batch', batch)
        self['valid_batch'] = hp[version].get('valid_batch', val_batch)
        self['w_g'] = 1
        self['awp_start'] = 2

        #请自己造训练测试集
        self['pretrain_file'] = 'data/data.csv'
        self['preval_file'] = 'data/preval.csv'
        self['train_file'] = 'data/raw.csv'
        self['valid_file'] = 'data/val_0.csv'
        self['test_file'] = 'data/preliminary_a_test.csv'
    
        self['input_l'] = 150
        self['output_l'] = 80
        self['n_token'] = 1560
        self['sos_id'] = 0
        self['pad_id'] = 1
        self['eos_id'] = 2
        
    def __init__(self, version, seed=0):
        self['lr'] = 1e-4
        self['model_dir'] = './checkpoint/%d'%version
        self['grid_dir'] = './grid/%d'%version
        self['pre_model_dir'] = './pretrain/%d'%version
        self['output_dir'] = './outputs/%d'%version
        
        self.version_config(version)
