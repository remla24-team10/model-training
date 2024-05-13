import os

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'