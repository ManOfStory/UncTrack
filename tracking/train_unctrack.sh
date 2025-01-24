# There are the detailed training settings for UncTrack-B and UncTrack-L.
# 1. download pretrained ConvMAE models (convmae_base.pth.pth/convmae_large.pth) at https://github.com/Alpha-VL/ConvMAE
# 2. set the proper pretrained convmae models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/unctrack(or unctrack_online)/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training UncTrack-B
# Stage1: train unctrack without PMN
python tracking/train.py --script unctrack --config baseline --save_dir /YOUR/PATH/TO/SAVE/UncTrack --mode multiple --nproc_per_node 8
## Stage2: train unctrack_online, i.e., PMN (prototype memory network)
# python tracking/train.py --script unctrack_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/UncTrack_Online --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training UncTrack-L
#python tracking/train.py --script unctrack --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/UncTrack-L --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL
#python tracking/train.py --script unctrack_online --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/UncTrack_L_Online --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL
