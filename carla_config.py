#TRAINING MODES
TRAIN_PLAY_MODE = 0 # Train=1 Play=0
GUARDAR_DATOS = 0   # Guardar=1 NO guardar=0
TRAIN_PLAY = ["PLAY", "TRAIN"]
WORKING_MODE_OPTIONS = ["WAYPOINTS_CARLA","WAYPOINTS_IMAGE","CNN_TRAJECTORY_BW","CNN_RGB","CNN_RGB_TRAJECTORY",
                        "CNN_SEMANTIC", "CNN_GRAYSCALE", "CNN_FLATTEN", "PRE_TRAINED_CNN"]
WORKING_MODE = "WAYPOINTS_CARLA"

#TRAINING STAGES
CARLA_MAP = "Town01"
TRAIN_MODE_OPTIONS = ["RANDOM", "STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "TURN_RIGHT_LEFT", "TURN_LEFT_RIGHT", "ALTERNATIVE"]
TRAIN_MODE = "RANDOM"

# PATHS
path2CARLA = "D:/app/WindowsNoEditor/" # PATH hasta carla se utiliza para limpiar el mapa
#path2carla = "/home/proyectosros/carla/carla/"
PRE_CNN_PATH = "PRE_CNN_models/PilotNet_2002m_BEV.model"
save_weights_path = "data/data_" + str(WORKING_MODE) + "/"
image_network = "NETWORKS/"

# actor_weights_file  = "data/data_" + str(WORKING_MODE) + "/" + str(TRAIN_MODE) + "_best_reward_actor.h5"
# critic_weights_file = "data/data_" + str(WORKING_MODE) + "/" + str(TRAIN_MODE) + "_best_reward_critic.h5"

# actor_weights_file  = "data/data_" + str(WORKING_MODE) + "_/TURN_RIGHT_LEFT_best_reward_actor.h5"
# critic_weights_file = "data/data_" + str(WORKING_MODE) + "_/TURN_RIGHT_LEFT_best_reward_critic.h5"
actor_weights_file  = "data/data_" + str(WORKING_MODE) + "_/_TURN_RIGHT_LEFT_150_actor.h5"
critic_weights_file = "data/data_" + str(WORKING_MODE) + "_/_TURN_RIGHT_LEFT_150_critic.h5"
# actor_weights_file  = "data/data_" + str(WORKING_MODE) + "/RANDOM_350_actor.h5"
# critic_weights_file = "data/data_" + str(WORKING_MODE) + "/RANDOM_350_critic.h5"


N_save_stats = 50

#IMAGE CONFIGURATION
IM_WIDTH_VISUALIZATION = 640
IM_HEIGHT_VISUALIZATION = 480
IM_WIDTH_CNN = 160
IM_HEIGHT_CNN = 60

#TRAIN PARAMETERS
tau = 0.001  # Target Network HyperParameter
lra = 0.0001  # Learning rate for Actor
lrc = 0.001  # Learning rate for Critic
FILTERS_CONV = 16
KERNEL_CONV = (5, 5)
POOL_SIZE = (3, 3)
POOL_STRIDES = (3, 3)
buffer_size = 100000
batch_size = 32
gamma = 0.99  # discount factor
hidden_units = (100, 400, 600)

#SIMULATION PARAMETERS
episodes_num = 8000
max_steps = 100000
SECONDS_PER_EPISODE = 10*70
SHOW_CAM = 0
SHOW_WAYPOINTS = 0
SHOW_CAM_RESIZE = 0
LINE_WIDHT = 2.5
LINE_TIME = 5
THRESHOLD = 0  # FLAG DE UMBRALIZACIÓN
DRAW_TRAJECTORY = 1 # NO PINTAR=0, PINTAR=1
IM_LAYERS = 1

# CARLA CAMERA OPTIONS
CAM_X = 3.5 #5 #0 #3.5
CAM_Z = 2.5 #20 #3.5 #2.5
CAM_PITCH = -40# -75.0 #-25.0 #-40.0
CAM_YAW = 0.0
CAM_ROLL = 0.0

#Vista de pájaro
BEV_PRE_CNN = 0


#WORKING TYPE SELECTION
WAYPOINTS = 'X'
#WAYPOINTS = 'XY'
dimension_vector_estado = 16
if WORKING_MODE == WORKING_MODE_OPTIONS[0]: # WAYPOINTS_CARLA
    state_dim = 30
    # CAM_X = 15
    # CAM_Z = 55
    # CAM_PITCH = -90.0
    # CAM_YAW = 0.0
    # CAM_ROLL = 0.0
    hidden_units = (300, 600)
    if WAYPOINTS =='XY':
        state_dim = 31
        dimension_vector_estado = state_dim
    elif WAYPOINTS == 'X':
        state_dim = 16
elif WORKING_MODE == WORKING_MODE_OPTIONS[1]: # WAYPOINTS_IMAGE
    DRAW_TRAJECTORY = 1
    state_dim = 17
    hidden_units = (300, 600)
    IM_WIDTH_VISUALIZATION = 640
    CAM_X = 1.5
    CAM_Z = 3.5
    CAM_PITCH = -25.0
    CAM_YAW = 0.0
    CAM_ROLL = 0.0
    LINE_TIME = 0
elif WORKING_MODE == WORKING_MODE_OPTIONS[2]: # CNN_TRAJECTORY_BW
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    THRESHOLD = 1                               # FLAG DE UMBRALIZACIÓN
    DRAW_TRAJECTORY = 1
    CAM_PITCH = -45.0
elif WORKING_MODE == WORKING_MODE_OPTIONS[3]: # CNN_RGB
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[4]: # CNN_RGB_TRAJECTORY
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    DRAW_TRAJECTORY = 1
elif WORKING_MODE == WORKING_MODE_OPTIONS[5]: # CNN_SEMANTIC
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 1                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[6]: # CNN_GRAYSCALE
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[7]:  # CNN_FLATTEN
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    THRESHOLD = 1                               # FLAG DE UMBRALIZACIÓN
    DRAW_TRAJECTORY = 1
    CAM_X = 2.0
    CAM_Z = 2.0
    CAM_PITCH = -20.0
    CAM_YAW = 0.0
    CAM_ROLL = 0.0
    IM_WIDTH_VISUALIZATION = 640
    IM_WIDTH_CNN = 11
    IM_HEIGHT_CNN = 11
    state_dim = IM_WIDTH_CNN * IM_HEIGHT_CNN
    LINE_WIDHT = 1.65
    if TRAIN_PLAY_MODE == 1:
        LINE_TIME = 10
elif WORKING_MODE == WORKING_MODE_OPTIONS[8]:       # PRE__TRAINED CNN
    if WAYPOINTS == 'XY':
        state_dim = 31
        dimension_vector_estado = state_dim
    elif WAYPOINTS == 'X':
        state_dim = 16
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    DRAW_TRAJECTORY = 1
    THRESHOLD = 0                               # FLAG DE UMBRALIZACIÓN
    IM_WIDTH_VISUALIZATION = 640 * 2
    # CAM_X = 0.0
    # CAM_Z = 3.5
    # CAM_PITCH = -25.0
    # CAM_YAW = 0.0
    # CAM_ROLL = 0.0

    CAM_X = 1.0
    CAM_Z = 1.8
    CAM_PITCH = -20.0
    CAM_YAW = 0.0
    CAM_ROLL = 0.0
    BEV_PRE_CNN = 1

