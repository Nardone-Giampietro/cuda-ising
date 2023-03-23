extern "C"
{
    __global__ void summ_final(float *x, float *total){
        __shared__ float psum[1024];
        int index = threadIdx.x;
        int inext;

        psum[index] = x[index];
        __syncthreads();

        inext = blockDim.x / 2;

        while (inext >= 1){
            if (index < inext){
                psum[index] = psum[index] + psum[index + inext];
            }
            inext = inext / 2;
            __syncthreads();
        }

        if (index == 0){
            total[0] = psum[0];
        }
    }
}

extern "C"
{
    __global__ void summ_partial(float *x, float *partial, int dim, int offset){
        __shared__ float psum[512];
        int index, inext;
        float thread_summ;

        index = blockIdx.x * blockDim.x + threadIdx.x;
        thread_summ = 0.0;
        for(int i = index; i < (dim - offset); i += blockDim.x * gridDim.x){
            thread_summ += x[i + offset];
        }

        index = threadIdx.x;
        psum[index] = thread_summ;
        __syncthreads();

        inext = blockDim.x / 2;
        while (inext >= 1){
            if (index < inext){
                psum[index] = psum[index] + psum[index + inext];
            }
            inext = inext / 2;
            __syncthreads();
        }

        if (index == 0){
            partial[blockIdx.x] = psum[0];            
        }
    }
}

extern "C"
{
    __global__ void summ_product_partial(float *x, float *partial, int dim, int offset){
        __shared__ float psum[256];
        int index, inext;
        float thread_sum;

        index = blockIdx.x * blockDim.x + threadIdx.x;
        thread_sum = 0.0;

        for(int i = index; i < (dim - offset); i += blockDim.x * gridDim.x){
            thread_sum += x[i] * x[i + offset];
        }

        index = threadIdx.x;
        psum[index] = thread_sum;
        __syncthreads();

        inext = blockDim.x / 2;
        while (inext >= 1){
            if (index < inext){
                psum[index] = psum[index] + psum[index + inext];
            }
            inext = inext / 2;
            __syncthreads();
        }

        if (index == 0){
            partial[blockIdx.x] = psum[0];
        }
    }
}