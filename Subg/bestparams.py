params = {
    "ppi_bp": {
        'aggr': 'mean',
        'batch_size': 48,
        'conv_layer': 4,
        'dropout': 0.25,
        'gn': True,
        'lr': 0.01,
        'pool': 'mean',
        'scalefreq': True,
        'z_ratio': 0.6
    },
    "hpo_metab": {
        'aggr': 'max',
        'batch_size': 16,
        'conv_layer': 3,
        'dropout': 0.35,
        'gn': False,
        'lr': 0.001,
        'pool': 'mean',
        'scalefreq': False,
        'z_ratio': 0.75
    },
    "hpo_neuro": {
        'aggr': 'mean',
        'batch_size': 48,
        'conv_layer': 4,
        'dropout': 0.35,
        'gn': False,
        'lr': 0.003,
        'pool': 'mean',
        'scalefreq': True,
        'z_ratio': 0.65
    },
    "em_user": {
        'aggr': 'sum',
        'batch_size': 12,
        'conv_layer': 1,
        'dropout': 0.5,
        'gn': True,
        'lr': 0.01,
        'pool': 'sum',
        'scalefreq': False,
        'z_ratio': 1
    },
    "component": {
        'aggr': 'sum',
        'batch_size': 64,
        'conv_layer': 1,
        'dropout': 0.1,
        'gn': True,
        'lr': 0.001,
        'pool': 'sum',
        'scalefreq': False,
        'z_ratio': 1
    },
    "density": {
        'aggr': 'sum',
        'batch_size': 64,
        'conv_layer': 1,
        'dropout': 0.1,
        'gn': True,
        'lr': 0.001,
        'pool': 'sum',
        'scalefreq': False,
        'z_ratio': 1
    },
    "coreness": {
        'aggr': 'sum',
        'batch_size': 64,
        'conv_layer': 1,
        'dropout': 0.1,
        'gn': True,
        'lr': 0.001,
        'pool': 'sum',
        'scalefreq': False,
        'z_ratio': 1
    },
    "cut_ratio": {
        'aggr': 'sum',
        'batch_size': 64,
        'conv_layer': 1,
        'dropout': 0.1,
        'gn': True,
        'lr': 0.01,
        'pool': 'sum',
        'scalefreq': False,
        'z_ratio': 1
    }
}