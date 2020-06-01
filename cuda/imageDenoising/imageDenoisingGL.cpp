/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



/*
 * This sample demonstrates two adaptive image denoising techniques:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter technique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

// CUDA utilities and system includes
#include <cuda_runtime.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageDenoising.h"

// includes, project
#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization

const char *sSDKsample = "CUDA ImageDenoising";

const char *filterMode[] =
{
    "Passthrough",
    "KNN method",
    "NLM method",
    "Quick NLM(NLM2) method",
    NULL
};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "image_passthru.ppm",
    "image_knn.ppm",
    "image_nlm.ppm",
    "image_nlm2.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_passthru.ppm",
    "ref_knn.ppm",
    "ref_nlm.ppm",
    "ref_nlm2.ppm",
    NULL
};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
//Source image on the host side
uchar4 *h_Src;
int imageW, imageH;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int  g_Kernel = 0;
bool    g_FPS = false;
bool   g_Diag = false;
StopWatchInterface *timer = NULL;

//Algorithms global parameters
const float noiseStep = 0.025f;
const float  lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float    lerpC = 0.2f;


const int frameN = 24;
int frameCounter = 0;


#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc   = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5
#define REFRESH_DELAY     10 //ms

void cleanup();

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "<%s>: %3.1f fps", filterMode[g_Kernel], ifps);

        // glutSetWindowTitle(fps);
        fpsCount = 0;

        //fpsLimit = (int)MAX(ifps, 1.f);
        sdkResetTimer(&timer);
    }
}

void runImageFilters(TColor *d_dst)
{
    switch (g_Kernel)
    {
        case 0:
            cuda_Copy(d_dst, imageW, imageH, texImage);
            break;

        case 1:
            if (!g_Diag)
            {
                cuda_KNN(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC, texImage);
            }
            else
            {
                cuda_KNNdiag(d_dst, imageW, imageH, 1.0f / (knnNoise * knnNoise), lerpC, texImage);
            }

            break;

        case 2:
            if (!g_Diag)
            {
                cuda_NLM(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC, texImage);
            }
            else
            {
                cuda_NLMdiag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC, texImage);
            }

            break;

        case 3:
            if (!g_Diag)
            {
                cuda_NLM2(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC, texImage);
            }
            else
            {
                cuda_NLM2diag(d_dst, imageW, imageH, 1.0f / (nlmNoise * nlmNoise), lerpC, texImage);
            }

            break;
    }

    getLastCudaError("Filtering kernel execution failed.\n");
}


void displayFunc(void)
{
    sdkStartTimer(&timer);
    TColor *d_dst = NULL;
    size_t num_bytes;

    if (frameCounter++ == 0)
    {
        sdkResetTimer(&timer);
    }

    runImageFilters(d_dst);

    if (frameCounter == frameN)
    {
        frameCounter = 0;

        if (g_FPS)
        {
            printf("FPS: %3.1f\n", frameN / (sdkGetTimerValue(&timer) * 0.001));
            g_FPS = false;
        }
    }

    sdkStopTimer(&timer);

    computeFPS();
}

void cleanup()
{
    free(h_Src);
    checkCudaErrors(CUDA_FreeArray());

    sdkDeleteTimer(&timer);
}

int main(int argc, char **argv)
{
    char *dump_file = NULL;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    pArgc = &argc;
    pArgv = argv;

    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char **)argv,
                                 "file", (char **) &dump_file);

        int kernel = 1;

        if (checkCmdLineFlag(argc, (const char **)argv, "kernel"))
        {
            kernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
        }

        runAutoTest(argc, argv, dump_file, kernel);
    }
    else
    {
        printf("[%s]\n", sSDKsample);

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            printf("[%s]\n", argv[0]);
            printf("   Does not explicitly support -device=n in OpenGL mode\n");
            printf("   To use -device=n, the sample must be running w/o OpenGL\n\n");
            printf(" > %s -device=n -qatest\n", argv[0]);
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }

        // First load the image, so we know what the size of the image (imageW and imageH)
        printf("Allocating host and CUDA memory and loading image file...\n");
        const char *image_path = sdkFindFilePath("portrait_noise.bmp", argv[0]);

        if (image_path == NULL)
        {
            printf("imageDenoisingGL was unable to find and load image file <portrait_noise.bmp>.\nExiting...\n");
            exit(EXIT_FAILURE);
        }

        LoadBMPFile(&h_Src, &imageW, &imageH, image_path);
        printf("Data init done.\n");

        findCudaDevice(argc, (const char **)argv);

        checkCudaErrors(CUDA_MallocArray(&h_Src, imageW, imageH));

    }

    TColor *d_dst = NULL;
    checkCudaErrors(cudaMalloc((void **)&d_dst, imageW*imageH*sizeof(TColor)));
    unsigned char *h_dst = NULL;
    h_dst = (unsigned char *)malloc(imageH*imageW*4);

    g_Kernel = 2;
    runImageFilters(d_dst);
    checkCudaErrors(cudaMemcpy(h_dst, d_dst, imageW*imageH*sizeof(TColor), cudaMemcpyDeviceToHost));

    printf("g_Kernel = %i\n", g_Kernel);

    SaveBMPFile(&h_Src, &imageW, &imageH, "bef.bmp");

    SaveBMPFile((uchar4 **)&h_dst, &imageW, &imageH, "aft.bmp");


    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

}
