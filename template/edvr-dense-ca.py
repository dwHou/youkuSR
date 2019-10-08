def load_arguments(parser):
    # Model specifications
    parser.add_argument('--model', default='edvr-dense-ca',
                        help='model name')
    parser.add_argument('--n_feats', type=int, default=128,
                        help='number of feature maps')
    parser.add_argument('--channel_growth', type=int, default=32,
                        help='number of channel growth')
    parser.add_argument('--n_groups', type=int, default=8,
                        help='number of deformable_groups')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='number of input frames')
    parser.add_argument('--n_densegroups', type=int, default=2,
                        help='number of dense groups')
    parser.add_argument('--n_resgroups', type=int, default=4,
                        help='number of residual groups')
    parser.add_argument('--n_cagroups', type=int, default=6,
                        help='number of channel atention groups')
    parser.add_argument('--n_resblocks', type=int, default=10,
                        help='number of residual blocks')
    parser.add_argument('--front_RBs', type=int, default=5,
                        help='number of residual groups before PCD Module')
    parser.add_argument('--back_RBs', type=int, default=40,
                        help='number of residual groups after TSA Module')
    return parser
