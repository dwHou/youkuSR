import argparse

def load_arguments(parser):
# Model specifications
    parser.add_argument('--model', default='rcan-dense',
                        help='model name')
    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--channel_growth', type=int, default=32,
                        help='channel size increment every resblock')
    parser.add_argument('--n_resblocks', type=int, default=10,
                        help='number of residual blocks')
    parser.add_argument('--n_resgroups', type=int, default=20,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    return parser
