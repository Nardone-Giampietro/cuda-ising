extern "C"
{
    __global__ void summ_final(double *x, double *total){
        __shared__ double psum[1024];
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
    __global__ void summ_partial(double *x, double *partial, int dim, int offset){
        __shared__ double psum[512];
        int index, inext;
        double thread_summ;

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
    __global__ void summ_product_partial(double *x, double *partial, int dim, int offset){
        __shared__ double psum[512];
        int index, inext;
        double thread_sum;

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

extern "C"
{
    __global__ void array_reduction(double *in_array, int dim, double *out_array){
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        int out_index;

        if (index < dim - 1){
            if ((index == 0) || (index % 2 == 0)){
                out_index = (int) index / 2;
                out_array[out_index] = (in_array[index] + in_array[index + 1]) / ((double) 2.0);
            }
        }
    }
}

extern "C"
{
    __global__ void resample(double *array_in, int dim, int *numbers, double *array_out){
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        int new_index;
        
        if (index < dim){
            new_index = numbers[index];
            array_out[index] = array_in[new_index];
        }
    }
}