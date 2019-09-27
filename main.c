#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

#include "libs/bitmap.h"

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5
// If you apply another kernel, remember not only to exchange
// the kernel but also the kernelFactor and the correct dimension.

int const sobelYKernel[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYKernelFactor = (float) 1.0;

int const sobelXKernel[] = {-1, -0, -1,
                            -2,  0, -2,
                            -1,  0, -1 , 0};
float const sobelXKernelFactor = (float) 1.0;


int const laplacian1Kernel[] = {  -1,  -4,  -1,
                                 -4,  20,  -4,
                                 -1,  -4,  -1};

float const laplacian1KernelFactor = (float) 1.0;

int const laplacian2Kernel[] = { 0,  1,  0,
                                 1, -4,  1,
                                 0,  1,  0};
float const laplacian2KernelFactor = (float) 1.0;

int const laplacian3Kernel[] = { -1,  -1,  -1,
                                  -1,   8,  -1,
                                  -1,  -1,  -1};
float const laplacian3KernelFactor = (float) 1.0;


//Bonus Kernel:

int const gaussianKernel[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1 };

float const gaussianKernelFactor = (float) 1.0 / 256.0;


// Helper function to swap bmpImageChannel pointers

void swapImageChannel(bmpImageChannel **one, bmpImageChannel **two) {
  bmpImageChannel *helper = *two;
  *two = *one;
  *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(unsigned char **out, unsigned char **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor) {
  unsigned int const kernelCenter = (kernelDim / 2);
  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = 0; x < width; x++) {
      int aggregate = 0;
      for (unsigned int ky = 0; ky < kernelDim; ky++) {
        int nky = kernelDim - 1 - ky;
        for (unsigned int kx = 0; kx < kernelDim; kx++) {
          int nkx = kernelDim - 1 - kx;

          int yy = y + (ky - kernelCenter);
          int xx = x + (kx - kernelCenter);
          if (xx >= 0 && xx < (int) width && yy >=0 && yy < (int) height)
            aggregate += in[yy][xx] * kernel[nky * kernelDim + nkx];
        }
      }
      aggregate *= kernelFactor;
      if (aggregate > 0) {
        out[y][x] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[y][x] = 0;
      }
    }
  }
}


void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  int ret = 0;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hi:";
  {
    char *endptr;
    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        goto graceful_exit;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          goto error_exit;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    goto error_exit;
  }
  input = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */

  //Starting mpi 
  MPI_Init(NULL, NULL);
  int worldSize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  int worldRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

  bmpImage *image=NULL;
  bmpImageChannel *imageChannel = newBmpImageChannel(0,0);

  if (worldRank == 0) // Only rank 0 need to create the image
  {
    /*
    Create the BMP image and load it from disk.
    */
    image = newBmpImage(0,0);

    if (image == NULL) {
      fprintf(stderr, "Could not allocate new image!\n");
    }

    if (loadBmpImage(image, input) != 0) {
      fprintf(stderr, "Could not load bmp image '%s'!\n", input);
      freeBmpImage(image);
      goto error_exit;
    }

    // Create a single color channel image. It is easier to work just with one color
    freeBmpImageChannel(imageChannel);
    imageChannel = newBmpImageChannel(image->width, image->height);
    if (imageChannel == NULL) {
      fprintf(stderr, "Could not allocate new image channel!\n");
      freeBmpImage(image);
      goto error_exit;
    }

    // Extract from the loaded image an average over all colors - nothing else than
    // a black and white representation
    // extractImageChannel and mapImageChannel need the images to be in the exact
    // same dimensions!
    // Other prepared extraction functions are extractRed, extractGreen, extractBlue
    if(extractImageChannel(imageChannel, image, extractAverage) != 0) {
      fprintf(stderr, "Could not extract image channel!\n");
      freeBmpImage(image);
      freeBmpImageChannel(imageChannel);
      goto error_exit;
    }
  }
  // To divide the image; all ranks need to know the size of what the get
  unsigned int *dimensions = malloc((worldSize+1) * sizeof(unsigned int));
  if (worldRank == 0)
  {
    for (int i = 0; i < worldSize; ++i)
    {
      // Divide image as equaly as possible
      dimensions[i] = imageChannel->height / worldSize; 

      // Divide the part that could not be done equaly among the first processes
      if (i < imageChannel->height % worldSize)
      {
        dimensions[i]++;
      }
      if (i < worldSize-1)  // If ghost border should be after section
      {
        dimensions[i]++;
      }

      if (i > 0)         // If ghost border should be before section
      {
        dimensions[i]++;
      }
    }

    dimensions[worldSize] = imageChannel->width;  // All can use the same width. Added to the end of the buffer
  }

  // Send the dimensions to each rank
  MPI_Bcast(dimensions, worldSize+1, MPI_UNSIGNED,
    0, MPI_COMM_WORLD
  );
  
  // allocate memory for the image section
  unsigned char *channelRawdata = malloc(dimensions[worldRank] * dimensions[worldSize] * sizeof(unsigned char));
  if (channelRawdata == NULL)
  {
    printf("Could not allocate memory for image chunk");
    goto error_exit;
  }

  int *sendCount = malloc(worldSize * sizeof(int));
  int *displacement = malloc(worldSize * sizeof(int));
  for (int i = 0; i < worldSize; ++i)
  {
    sendCount[i] = dimensions[i] * dimensions[worldSize];
    if (i > 0)  // If not rank 0's offset
    {
      // calculate displacement with all previous displacements and size of previouse chunk size minus the ghost borders.
      displacement[i] = displacement[i-1] + (dimensions[i-1]-2) * dimensions[worldSize];
    }else{
      displacement[0] = 0;
    }
  }
    
  // Divide the image
  MPI_Scatterv(imageChannel->rawdata, sendCount, displacement, MPI_UNSIGNED_CHAR,
      channelRawdata, dimensions[worldRank] * dimensions[worldSize], MPI_UNSIGNED_CHAR,
      0, MPI_COMM_WORLD
  );

  // Create a bmpImageChannel from the imageChunk for computation
  bmpImageChannel *imageChannelBmp = newBmpImageChannel(dimensions[worldSize], dimensions[worldRank]);
  memcpy(imageChannelBmp->rawdata, channelRawdata, dimensions[worldSize] * dimensions[worldRank] * sizeof(unsigned char));
  free(dimensions); free(channelRawdata);

  //Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x])

  bmpImageChannel *processImageChannel = newBmpImageChannel(imageChannelBmp->width, imageChannelBmp->height);
  for (unsigned int i = 0; i < iterations; i ++) {
    applyKernel(processImageChannel->data,
                imageChannelBmp->data,
                imageChannelBmp->width,
                imageChannelBmp->height,
                (int *)laplacian1Kernel, 3, laplacian1KernelFactor
 //               (int *)laplacian2Kernel, 3, laplacian2KernelFactor
 //               (int *)laplacian3Kernel, 3, laplacian3KernelFactor
 //               (int *)gaussianKernel, 5, gaussianKernelFactor
                );
    swapImageChannel(&processImageChannel, &imageChannelBmp);
  }
  freeBmpImageChannel(processImageChannel);

 // printf("(%d/%d) Waiting...\n",worldRank, getpid() );
 // sleep(10);
  // Create a buffer for the  created image
  unsigned char *resultImageRawdata = NULL; 
  if (worldRank == 0) // Only root needs space to store the actual result
  {
      resultImageRawdata = malloc(imageChannel->width * imageChannel->height * sizeof(unsigned char));
  }

  // Remove border before gather
  int ghostBefore = worldRank != 0 ? 1:0;
  int ghostAfter = worldRank != worldSize - 1 ? 1:0;
  int rankSendCount = (imageChannelBmp->height - (ghostBefore + ghostAfter)) * imageChannelBmp->width;

  int *recieveCount = NULL;
  if (worldRank == 0)   // Only root need recieveCount and displacement
  {
    recieveCount = malloc(worldSize * sizeof(int));
    memcpy(recieveCount, sendCount, worldSize * sizeof(int));
    free(sendCount);
    for (int i = 0; i < worldSize; ++i)
    {
      if (i < worldSize -1)  // Remove ghost border after section
      {
        recieveCount[i] -= imageChannel->width;
      }

      if (i > 0)         // If ghost border should be before section
      {
        recieveCount[i] -= imageChannel->width;
        displacement[i] = recieveCount[i-1] + displacement[i-1];
     
      }else{
        displacement[0] = 0;
      }
    }

   
  }
  // Reassemble the image data
  MPI_Gatherv(imageChannelBmp->data[ghostBefore], rankSendCount, MPI_UNSIGNED_CHAR,
      resultImageRawdata, recieveCount, displacement, MPI_UNSIGNED_CHAR,
      0, MPI_COMM_WORLD
  );
  freeBmpImageChannel(imageChannelBmp); free(recieveCount); free(displacement);

  if (worldRank == 0) // Only rank 0 can reasemble the image
  {
    // Create a bmpImageChannel for the result
    bmpImageChannel *imageChannelResult = newBmpImageChannel(image->width, image->height);
    memcpy(imageChannelResult->rawdata, resultImageRawdata, image->width * image->height * sizeof(unsigned char));
    free(resultImageRawdata);

    // Map our single color image back to a normal BMP image with 3 color channels
    // mapEqual puts the color value on all three channels the same way
    // other mapping functions are mapRed, mapGreen, mapBlue
    if (mapImageChannel(image, imageChannelResult, mapEqual) != 0) {
      fprintf(stderr, "Could not map image channel!\n");
      freeBmpImage(image);
      freeBmpImageChannel(imageChannel);
      freeBmpImageChannel(imageChannelResult);
      goto error_exit;
    }
    freeBmpImageChannel(imageChannel);
    freeBmpImageChannel(imageChannelResult);

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0) {
      fprintf(stderr, "Could not save output to '%s'!\n", output);
      freeBmpImage(image);
      goto error_exit;
    };
    
  }
  MPI_Finalize();


graceful_exit:
  ret = 0;
error_exit:
  if (input)
    free(input);
  if (output)
    free(output);
  return ret;
};
