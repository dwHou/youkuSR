def load_arguments(parser):
    # Model specifications
    parser.add_argument('--model', default='edvr',
                        help='model name')
    parser.add_argument('--n_feats', type=int, default=128,
                        help='number of feature maps')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--n_groups', type=int, default=8,
                        help='number of deformable_groups')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='number of input frames')
    parser.add_argument('--front_RBs', type=int, default=5,
                        help='number of residual groups before PCD Module')
    parser.add_argument('--back_RBs', type=int, default=40,
                        help='number of residual groups after TSA Module')
    parser.add_argument('--n_resgroups', type=int, default=6,
                        help='number of residual groups')
    parser.add_argument('--n_resblocks', type=int, default=10,
                        help='number of residual blocks')
    return parser
