echo pqnet
./fid_score.py ../../experiments/results/gt_test/test64 ../../experiments/results/pqnet --gpu 0 --task thres
echo housegan
./fid_score.py ../../experiments/results/gt_test/test64 ../../experiments/results/housegan --gpu 0 --task thres
echo roofgan
./fid_score.py ../../experiments/results/gt_test/test64 ../../experiments/results/roofgan --gpu 0 --task graph
