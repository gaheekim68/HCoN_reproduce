# python run.py --gpu_id 0 --dataname coauthor_dblp --data_dir ../dataset/coauthorship/dblp --raw_data_dir ../edgnn-hypergraph-dataset/coauthorship/dblp
# python run.py --gpu_id 0 --dataname coauthor_cora --data_dir ../dataset/coauthorship/cora --raw_data_dir ../edgnn-hypergraph-dataset/coauthorship/cora
# python run.py --gpu_id 0 --dataname citeseer --data_dir ../dataset/cocitation/citeseer --raw_data_dir ../edgnn-hypergraph-dataset/cocitation/citeseer
# python run.py --gpu_id 0 --dataname cora --data_dir ../dataset/cocitation/cora --raw_data_dir ../edgnn-hypergraph-dataset/cocitation/cora
# python run.py --gpu_id 0 --dataname senate-committees --data_dir ../dataset/senate-committees --raw_data_dir ../edgnn-hypergraph-dataset/senate-committees

#!/bin/bash
for lr in 0.01 0.001 0.005
do
  for wd in 1e-4 1e-5 1e-6
  do
    for hd in 64 128 256
    do
      echo "Running: lr=$lr, wd=$wd, hd=$hd"
      python run.py \
        --dataname citeseer \
        --data_dir ../dataset/cocitation/citeseer \
        --raw_data_dir ../edgnn-hypergraph-dataset/cocitation/citeseer \
    done
  done
done
