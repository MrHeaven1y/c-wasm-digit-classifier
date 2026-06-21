#pragma once

typedef struct {
    int num;
    int rows;
    int cols;
    unsigned char* images; // num × 784
    unsigned char* labels; // num
} MNIST;

MNIST load_mnist(const char* img_path, const char* label_path);
void free_mnist(MNIST* m);
