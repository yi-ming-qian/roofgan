echo pqnet
python ./fid_score.py ../../experiments/results/gt_test/test64 ../../experiments/results/pqnet --gpu 0 --task thres
echo housegan
python ./fid_score.py ../../experiments/results/gt_test/test64 ../../experiments/results/housegan --gpu 0 --task thres
echo roofgan
python ./fid_score.py ../../experiments/results/gt_test/test64 ../../experiments/results/roofgan --gpu 0 --task graph
