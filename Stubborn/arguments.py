import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Stubborn')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--sem_gpu_id', type=int, default=-1,
                        help='GPU ID')

    parser.add_argument('-d', '--dump_location', type=str, default="./tmp/",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--checkpt', type=str, default="./Stubborn/rednet_semmap_mp3d_tuned.pth",
                        help='path to rednet models')

    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--rednet_channel', type=int, default=20,
                        help="""rednet channel""")
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:120)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500, # originally 500
                        help="""Maximum episode length""")
    parser.add_argument("--task_config", type=str,
                        default="tasks/objectnav_gibson.yaml",
                        help="path to config yaml containing task information")

    parser.add_argument('--camera_height', type=float, default=0.88,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")


    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local policy
                                between each global step""")

    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=6) # originally 6 (don't forget the change goal threshold)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=7200) # the global downscaling also need to be changed
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.20)
    parser.add_argument(
        "--evaluation", type=str, required=False, choices=["local", "remote"]
    )
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('--timestep_limit', type=int, default=500)
    parser.add_argument('--log_path', type=str,
                        default="./log.pickle",
                        help="directory to store log"
                        )
    parser.add_argument('--change_goal_threshold', type = float, default = 240) # originally 240

    parser.add_argument('--record_frames', type = int, default = 2)
    parser.add_argument('--record_angle',type = int, default = 2)
    parser.add_argument('--grid_resolution',type = int, default = 24)
    parser.add_argument('--record_conflict',type = int, default = 1)
    parser.add_argument('--magnify_goal_when_hard',type = int, default = 100) #originally 100
    parser.add_argument("--move_forward_after_stop",type = int, default = 1) #originally 1
    parser.add_argument("--small_collision_map_for_goal",type = int, default = 0)

    # for data collection purposes. Use 0 to turn off
    # use 1 to turn on
    parser.add_argument("--no_stop",type = int, default = 0)
    parser.add_argument('--use_gt_mask',type = int, default = 0)
    parser.add_argument('--detect_stuck',type = int, default = 0)
    parser.add_argument('--only_explore',type = int, default = 0)
    parser.add_argument('--exclude_current_scene',type = int, default = 0)

    # parse arguments
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()



    return args
