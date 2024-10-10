bparams = {
    "label": {
        "NDC-classes": {
            'batch_size': 96,
            'dp0': 0.1,
            'hid_dim': 64, #112,
            'layerwise_label': True,
            'lr': 0.005,
            'num_layer': 4,
            'scalefreq': True,
            'share_label': True,
            'share_lin': False,
            'use_act': False
        },
        "DAWN": {
            'batch_size': 256, #96,
            'dp0': 0.1,
            'hid_dim': 32,#64, #112,
            'layerwise_label': True,
            'lr': 0.005,
            'num_layer': 4,
            'scalefreq': True,
            'share_label': False, #True,
            'share_lin': False,
            'use_act': False
        },
        "email-Eu": {
            'batch_size': 96,
            'dp0': 0.1,
            'hid_dim': 64, #112,
            'layerwise_label': True,
            'lr': 0.005,
            'num_layer': 4,
            'scalefreq': True,
            'share_label': True,
            'share_lin': False,
            'use_act': False
        },
        "NDC-substances": {
            'batch_size': 96,
            'dp0': 0.1,
            'hid_dim': 64, #112,
            'layerwise_label': True,
            'lr': 0.005,
            'num_layer': 4,
            'scalefreq': True,
            'share_label': True,
            'share_lin': False,
            'use_act': False
        },
        "tags-ask-ubuntu": {
            'batch_size': 96,
            'dp0': 0.1,
            'hid_dim': 64, #112,
            'layerwise_label': True,
            'lr': 0.005,
            'num_layer': 4,
            'scalefreq': True,
            'share_label': True,
            'share_lin': False,
            'use_act': False
        },
    },
    "plabel": {
        "NDC-classes": {
            'batch_size': 96,
            'dp0': 0.1,
            'hid_dim': 64, # 128
            'layerwise_label': True,
            'lr': 0.003,
            'num_layer': 4,
            'scalefreq': True,
            'share_label': False,
            'share_lin': True,
            'use_act': False,
        },
        "DAWN": {
            'batch_size': 256, #96,
            'dp0': 0.1,
            'hid_dim': 32, #64, 
            'layerwise_label': True,
            'lr': 0.005,
            'num_layer': 4,
            'scalefreq': True,
            'share_label': False,
            'share_lin': False,
            'use_act': False
        }
    }
}