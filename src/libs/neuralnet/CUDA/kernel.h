extern "C" cudaTextureObject_t texImage;

extern "C" void cuda_Copy(TColor *d_dst, int imageW, int imageH, cudaTextureObject_t texImage);
