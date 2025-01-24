
# Different test settings for UncTrack-B, UncTrack-L on LaSOT/TrackingNet/GOT10K/UAV123/OTB100/NAT2021
# First, put your trained UncTrack online models on SAVE_DIR/models directory.
# Then,uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH.

##########-------------- UncTrack-B -----------------##########
### LaSOT test and evaluation
#python tracking/test.py unctrack_online baseline --dataset lasot --threads 16 --num_gpus 4 --params__model unctrack_base_online.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline

### TrackingNet test
# python tracking/test.py unctrack_online baseline --dataset trackingnet --threads 16 --num_gpus 4 --params__model unctrack_base_online.pth.tar

### GOT10k test
# python tracking/test.py unctrack_online baseline --dataset got10k_test --threads 16 --num_gpus 4 --params__model unctrack_base_online.pth.tar --params__search_area_scale 4.5

### UAV123
# python tracking/test.py unctrack_online baseline --dataset uav --threads 16 --num_gpus 4 --params__model unctrack_base_online.pth.tar --params__search_area_scale 4.5
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline

### OTB100
#python tracking/test.py unctrack_online baseline --dataset otb --threads 16 --num_gpus 8 --params__model unctrack_base_online.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline

### NAT2021
python tracking/test.py unctrack_online baseline --dataset nat2021 --threads 16 --num_gpus 4 --params__model unctrack_base_online.pth.tar --params__search_area_scale 4.5
python tracking/analysis_results.py --dataset_name nat2021 --tracker_param baseline


##########-------------- UncTrack-L -----------------##########
### LaSOT test and evaluation
#python tracking/test.py unctrack_online baseline_large --dataset lasot --threads 16 --num_gpus 4 --params__model unctrack_large_online.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large

### TrackingNet test
# python tracking/test.py unctrack_online baseline_large --dataset trackingnet --threads 16 --num_gpus 4 --params__model unctrack_large_online.pth.tar

### GOT10k test
# python tracking/test.py unctrack_online baseline_large --dataset got10k_test --threads 16 --num_gpus 4 --params__model unctrack_large_online_got.pth.tar --params__search_area_scale 4.5

### UAV123
# python tracking/test.py unctrack_online baseline_large--dataset uav --threads 16 --num_gpus 4 --params__model unctrack_large_online.pth.tar --params__search_area_scale 4.5
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_large

### OTB100
#python tracking/test.py unctrack_online baseline_large--dataset otb --threads 16 --num_gpus 8 --params__model unctrack_large_online.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_large

### NAT2021
#python tracking/test.py unctrack_online baseline_large --dataset nat2021 --threads 16 --num_gpus 4 --params__model unctrack_large_online.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name nat2021 --tracker_param baseline_large


