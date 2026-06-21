#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "../include/mnist.h"

static int read_be_int(FILE* f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
}

MNIST load_mnist(const char* img_path, const char* label_path) {
    MNIST m;

    FILE* fi = fopen(img_path, "rb");
    FILE* fl = fopen(label_path, "rb");

    if (!fi || !fl) {
        printf("MNIST files not found\n");
        exit(1);
    }

    read_be_int(fi); // magic
    m.num = read_be_int(fi);
    m.rows = read_be_int(fi);
    m.cols = read_be_int(fi);

    read_be_int(fl); // magic
    read_be_int(fl); // num labels

    m.images = malloc(m.num * m.rows * m.cols);
    m.labels = malloc(m.num);

    fread(m.images, 1, m.num * m.rows * m.cols, fi);
    fread(m.labels, 1, m.num, fl);

    fclose(fi);
    fclose(fl);

    return m;
}

void free_mnist(MNIST* m) {
    free(m->images);
    free(m->labels);
}
