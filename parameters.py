#! /usr/bin/env python2
import sys

default_vals = [
    ('debug', 2, 'Debugging level'),

    #
    # Parameters controlling feature extraction for initial alignment
    #
    ('use_klt', 0, 'Use KLT features? Otherwise, features from initial flow are used (set to 1 or 0)'),
    ('feature_sampling', 10, 'Which distance to sample feature from when sampling from flow'),
    ('featurelocations_nms', 0, 'Whether we want to refine the feature locations using non-maximum suppresion on image gradients.'),

    # Alignment section
    ('alignment_ransac_threshold', 2.0, 'RansacReprojThreshold for initial homography estimation'),
    ('alignment_error_type', 3, 'Error type to use when refining alignment. 0=quadratic, 1=charbonnier, 2=lorentzian, 3=Geman-Mcclure'),
    ('alignment_ransac', 0, 'Whether to use RANSAC for initial alignment (=1) or the algorithm described in H&Z (=0)'),
    ('refine_coplanarity', 1, 'Perform coplanarity refinement step'),


    # Parameters controlling the rigidity estimation from motion
    ('rigidity_use_structure', 1, 'Rigidity estimation: Incorporate structure consistency'),
    ('rigidity_weight_cnn', 0.25, 'Rigidity estimation: Weight of the CNN term.'),
    ('rigidity_sigma_structure', 1.0, 'Rigidity estimation: Sigma for rigidity from structure difference'),
    ('rigidity_sigma_direction', 1.0, 'Rigidity estimation: Sigma for rigiditiy from direction'),

    # Parameters controlling occlusion computation
    ('occlusion_reasoning', 1, 'Do occlusion reasoning'),
    ('occlusion_threshold', 2.0, 'Consistency threshold'),

    # Parameters controlling initialization
    ('nonlinear_structure_initialization', 1, 'Refine H, b before optimizing the model'),
    ('scale_structure', 1, 'Scale structure. 0=No scaling, 1=Scaling so that MAD=1, 2=Scaling so that STD=1.'),
 

    # Parameters for variational refinement
    ('override_optimization', 0, 'Override optimization, return init instead.'),
    ('variational_lambda_1storder', 0.5, 'Variational structure refinement: 1st order regularization weight'), #0.05
    ('variational_lambda_2ndorder', 5.0, 'Variational structure refinement: 2nd order regularization weight'), #5.0
    ('variational_lambda_consistency', 0.0, 'Variational structure refinement: Consistency weight'), #0.001
    ('variational_N_outer', 5, 'Variational structure refinement: Number of outer iterations'),
    ('variational_N_inner', 1, 'Variational structure refinement: Number of inner iterations'),
    ('variational_scale_factor', 0.5, 'Variational structure refinement: Scale factor between pyramid levels'),
    ('variational_N_scales', 1, 'Variational structure refinement: Number of pyramid levels'),
    ('variational_last_scale', 0, 'Variational structure refinement: Bottom pyramid layer'),
    ('variational_dataterm_grayscale', 1, 'Convert images to grayscale before computing the data term'),
    ('variational_min_weight', 0.0, 'Minimal value for weights (gradient and image-based weights) when computing the sparse problem.'),


    ('debug_save_frames', 0, 'Save intermediate results as images'),
    ('debug_compute_figure', 0, 'Compute output figure from paper'),
    ('tempdir', '.', 'Output directory'),
    ('debug_use_rigidity_gt', 0, 'Use GT rigidity instead of estimation'),

    ]


preset_parameters = {
        }



def add_parameters_argparse(parser):
    for param, default, helptext in (default_vals):
        parser.add_argument('--{}'.format(param), type=type(default), default=default, help=helptext)




def get_parameters(params=None,preset=None,do_print=False):
    # Read default values
    p = dict(default_vals)

    if preset is not None:
        assert(preset_parameters.has_key(preset))
        print('*** LOADING PRESET {} ***'.format(preset))
        params_preset = preset_parameters[preset]
        for k,v in params_preset.items():
            assert(p.has_key(k))
            p[k] = v

    # Read values from command line
    # For these, convert parameters to int.
    params_int = ['NC', 'image_blur', 'SUBLAYER_NC', 'n_models']
    for k,v in p.items():
        prm = '-'+k
        if not prm in sys.argv:
            # Value was not found
            continue
        else:
            # Read in value
            v_ = sys.argv[sys.argv.index(prm)+1]
            if k == 'features' or k == 'adapt_size_mismatch':
                pass
            elif k in params_int:
                v_ = int(float(v_))
            else:
                v_ = float(v_)
            p[k] = v_
    
    if params is not None:
        # Read values from given parameters -- those override all previous.
        params_ = dict(params)
        for k,v in params_.items():
            if do_print:
                print('Replacing entry {}. Old: {}, new: {}'.format(k,p[k],v))
            p[k] = v

    if do_print:
        print('')
        print('[PARAMETERS MAIN]')
        print('')
        for k2 in sorted(p.keys()):
            v2 = p[k2]
            print('\t{0}: \t{1}'.format(k2,v2))

    return p



#def get_sublayer_parameters(params=None):
    #p = dict(get_parameters(params),do_print=False)

    #for k,v in p.items():
        #if k.startswith('SUBLAYER_'):
            #k2 = k.strip('SUBLAYER_')
            #print('k : {} :: k2 : {}'.format(k,k2))
            #assert(p.has_key(k2))
            #p[k2] = v

    #print('')
    #print('[PARAMETERS SUBLAYER]:')
    #print('')
    #for k in sorted(p.keys()):
        #v2 = p[k]
        #print('\t{0}: \t{1}'.format(k,v2))


    #return p

