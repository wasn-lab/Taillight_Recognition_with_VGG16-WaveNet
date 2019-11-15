set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3
	-gencode arch=compute_30,code=sm_30
	-gencode arch=compute_35,code=sm_35
	-gencode arch=compute_50,code=[sm_50,compute_50]
	-gencode arch=compute_52,code=[sm_52,compute_52]
	-gencode arch=compute_61,code=sm_61
	-gencode arch=compute_62,code=sm_62
	--keep  # keep intermediate files for generating coverage reports.
)

