#include <stdio.h>
#include <string.h>
#include <assert.h>

// TODO Check image resolution of dslr and update it here.
#define MAX_IMAGE_WIDTH   1920
#define MAX_IMAGE_HEIGHT  1080
#define MAX_SIZE          19
#define uint unsigned int

extern "C"{

void AdaptiveThreshold(unsigned short width,
                       unsigned short height,
                       unsigned short size,
                       int *src,
                       unsigned char *dst);
}

