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


#include <stdio.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(1)

typedef struct
{
    short type;
    int size;
    short reserved1;
    short reserved2;
    int offset;
} BMPHeader;

typedef struct
{
    int size;
    int width;
    int height;
    short planes;
    short bitsPerPixel;
    unsigned compression;
    unsigned imageSize;
    int xPelsPerMeter;
    int yPelsPerMeter;
    int clrUsed;
    int clrImportant;
} BMPInfoHeader;



//Isolated definition
typedef struct
{
    unsigned char x, y, z, w;
} uchar4;



extern "C" void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name)
{
    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x, y;

    FILE *fd;


    printf("Loading %s...\n", name);

    if (!(fd = fopen(name,"rb")))
    {
        printf("***BMP load error: file access denied***\n");
        exit(EXIT_SUCCESS);
    }

    fread(&hdr, sizeof(hdr), 1, fd);

    if (hdr.type != 0x4D42)
    {
        printf("***BMP load error: bad file format***\n");
        exit(EXIT_SUCCESS);
    }

    fread(&infoHdr, sizeof(infoHdr), 1, fd);

    if (infoHdr.bitsPerPixel != 24)
    {
        printf("***BMP load error: invalid color depth***\n");
        exit(EXIT_SUCCESS);
    }

    if (infoHdr.compression)
    {
        printf("***BMP load error: compressed image***\n");
        exit(EXIT_SUCCESS);
    }

    *width  = infoHdr.width;
    *height = infoHdr.height;
    *dst    = (uchar4 *)malloc(*width **height * 4);

    printf("type: %i\n", hdr.type);
    printf("size: %i\n", hdr.size);
    printf("reserved1: %i\n", hdr.reserved1);
    printf("reserved2: %i\n", hdr.reserved2);
    printf("offset: %i\n", hdr.offset);

    printf("BMP size: %u\n", infoHdr.size);
    printf("BMP width: %u\n", infoHdr.width);
    printf("BMP height: %u\n", infoHdr.height);
    printf("BMP planes: %u\n", infoHdr.planes);
    printf("BMP bitsPerPixel: %u\n", infoHdr.bitsPerPixel);
    printf("BMP compression: %u\n", infoHdr.compression);
    printf("BMP imageSize: %u\n", infoHdr.imageSize);
    printf("BMP xPelsPerMeter: %u\n", infoHdr.xPelsPerMeter);
    printf("BMP yPelsPerMeter: %u\n", infoHdr.yPelsPerMeter);
    printf("BMP clrUsed: %u\n", infoHdr.clrUsed);
    printf("BMP clrImportant: %u\n", infoHdr.clrImportant);

    fseek(fd, hdr.offset - sizeof(hdr) - sizeof(infoHdr), SEEK_CUR);

    for (y = 0; y < infoHdr.height; y++)
    {
        for (x = 0; x < infoHdr.width; x++)
        {
            (*dst)[(y * infoHdr.width + x)].z = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].y = fgetc(fd);
            (*dst)[(y * infoHdr.width + x)].x = fgetc(fd);
        }

        for (x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++)
            fgetc(fd);
    }

    if (ferror(fd))
    {
        printf("***Unknown BMP load error.***\n");
        free(*dst);
        exit(EXIT_SUCCESS);
    }
    else
        printf("BMP file loaded successfully!\n");

    fclose(fd);
}

#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

extern "C" void SaveBMPFile(uchar4 **dst, int *width, int *height, const char *name) {
    BMPHeader hdr;
    BMPInfoHeader infoHdr;
    int x, y;

    int fd = open(name
    , O_RDWR | O_CREAT
    , S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);

    hdr.type = 0x4D42;
    // hdr.size;
    // hdr.reserved1;
    // hdr.reserved2;
    hdr.offset = 54;

    ssize_t cv = write(fd, &hdr, sizeof(hdr));

    infoHdr.width = *width;
    infoHdr.height = *height;
    infoHdr.bitsPerPixel = 24;
    infoHdr.compression = 0;
    infoHdr.imageSize = 0;
    //test
    infoHdr.size = 40;
    infoHdr.planes = 1;
    // infoHdr.xPelsPerMeter = 3779;
    // infoHdr.yPelsPerMeter = 3779;
    // infoHdr.clrUsed = 0;
    // infoHdr.clrImportant = 0;
    cv += pwrite(fd, &infoHdr, sizeof(infoHdr), cv);

    unsigned char zero = '\0';
    uchar4 *dst_ptr;
    for (y = 0; y < *height; y++)
    {
        for (x = 0; x < *width; x++)
        {
            dst_ptr = (*dst) + (y * infoHdr.width + x);
            cv += pwrite(fd, &dst_ptr->z, 1, cv);
            cv += pwrite(fd, &dst_ptr->y, 1, cv);
            cv += pwrite(fd, &dst_ptr->x, 1, cv);
        }

        // for (x = 0; x < (4 - (3 * *width) % 4) % 4; x++)
        //     cv += pwrite(fd, &zero, 1, cv);
    }

    // write(fd, dst, strlen((const char*)dst));
    close(fd);
}