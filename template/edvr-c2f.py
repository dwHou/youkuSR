def load_arguments(parser):
    # Model specifications
    parser.add_argument('--model', default='edvr-c2f',
                        help='model name')
    parser.add_argument('--n_feats', type=int, default=128,
                        help='number of feature maps')
    parser.add_argument('--n_groups', type=int, default=8,
                        help='number of deformable_groups')
    parser.add_argument('--n_frames', type=int, default=9,
                        help='number of input frames')
    parser.add_argument('--n_resgroups', type=int, default=10,
                        help='number of residual groups')
    parser.add_argument('--n_cagroups', type=int, default=0,
                        help='number of channel atention groups')
    parser.add_argument('--n_resblocks', type=int, default=10,
                        help='number of residual blocks')
    parser.add_argument('--n_resblocks_e', type=int, default=10,
                        help='number of residual blocks')
    parser.add_argument('--n_cagroups_e', type=int, default=10,
                        help='number of channel atention groups')
    parser.add_argument('--front_RBs', type=int, default=10,
                        help='number of residual groups before PCD Module')
    parser.add_argument('--back_RBs', type=int, default=0,
                        help='number of residual groups after TSA Module')
    return parser
