rm *.nsys-rep; nsys profile --sample process-tree --nvtx-capture domain -t nvtx,cuda,cudnn,cublas,osrt -x true -f true -o profile ./simpleCUBLAS
nsys stats profile.nsys-rep
