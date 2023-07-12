#nsys profile --sample process-tree --nvtx-capture domain -t nvtx,cuda,cudnn,cublas,osrt -x true -f true -o profile_cublas ./simpleCUBLAS
sudo /usr/local/cuda/bin/ncu -f -o profile --cache-control none --metrics=lts__throughput.avg.pct_of_peak_sustained_elapsed  ./simpleCUBLAS
echo $PWD/profile.ncu-rep
