#from utils import seed_everything
import json

with open('/Users/dgonzalez/Documents/dissertation/activity_to_encoding_map.txt', 'r') as f:
    one_hot_labels = json.loads(f.read())
    
ONE_HOT_LABELS = {int(key):one_hot_labels[key].replace('activity_','') for key in one_hot_labels.keys()}
#print (ONE_HOT_LABELS)

DATASET = 'PAMAP2'
DATA_DIR = '/Users/dgonzalez/Documents/dissertation/'

TIMESTEPS = 172
FEATURES = 40

DEVICE = 'cpu' #"cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
PIN_MEMORY = True

NUM_CLASSES = 18
BATCH_SIZE = 16
NUM_EPOCHS = 50

LEARNING_RATE = .0001
WEIGHT_DECAY = 0

LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = DATA_DIR+"checkpoint.pth.tar"

LABELS_SET = ['lying', 
                'sitting', 
                'standing', 
                'walking',
                'running', 
                'cycling', 
                'Nordic walking', 
                'ascending stairs', 
                'descending stairs', 
                'vacuum cleaning', 
                'ironing']

OPEN_SET = False
if OPEN_SET:
    EXPERIMENT = 'TSCResNet_OpenSet'
    LABELS = ['lying', 
              'sitting', 
              'standing', 
              'walking',
              'running', 
              'cycling', 
              'Nordic walking', 
              'ascending stairs', 
              'descending stairs', 
              'vacuum cleaning', 
              'ironing'][:]
else:
    EXPERIMENT = 'TSCResNet'
    LABELS = 'NA'