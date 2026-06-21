#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/tensor.h"
#include "../include/mnist.h"

/* --- 1. Model Definition (Must match main.c) --- */
typedef struct {
    Tensor *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
} MLP;

MLP init_mlp(GraphContext* ctx) {
    // We initialize with dummy random weights, then overwrite them with the file
    return (MLP){
        RandParam(ctx, 784, 256), Param(ctx, calloc(256, sizeof(double)), 1, 256),
        RandParam(ctx, 256, 128), Param(ctx, calloc(128, sizeof(double)), 1, 128),
        RandParam(ctx, 128, 64),  Param(ctx, calloc(64, sizeof(double)), 1, 64),
        RandParam(ctx, 64, 10),   Param(ctx, calloc(10, sizeof(double)), 1, 10)
    };
}

Tensor* mlp_forward(GraphContext* ctx, MLP* m, Tensor* x) {
    Tensor* h1 = Relu(ctx, Add(ctx, MatMul(ctx, x, m->W1), m->b1));
    Tensor* h2 = Relu(ctx, Add(ctx, MatMul(ctx, h1, m->W2), m->b2));
    Tensor* h3 = Relu(ctx, Add(ctx, MatMul(ctx, h2, m->W3), m->b3));
    return Add(ctx, MatMul(ctx, h3, m->W4), m->b4);
}

/* --- 2. Load Helpers --- */
void load_tensor_bin(FILE* f, Tensor* t) {
    int rows, cols;
    fread(&rows, sizeof(int), 1, f);
    fread(&cols, sizeof(int), 1, f);
    fread(t->storage->data, sizeof(double), rows * cols, f);
}

int load_model_bin(MLP* m, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    load_tensor_bin(f, m->W1); load_tensor_bin(f, m->b1);
    load_tensor_bin(f, m->W2); load_tensor_bin(f, m->b2);
    load_tensor_bin(f, m->W3); load_tensor_bin(f, m->b3);
    load_tensor_bin(f, m->W4); load_tensor_bin(f, m->b4);
    fclose(f);
    return 1;
}

int argmax(Tensor* t) {
    double max_val = -1e9;
    int max_idx = 0;
    for (int i = 0; i < t->cols; i++) {
        if (t->storage->data[i] > max_val) {
            max_val = t->storage->data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/* --- 3. Main Test Loop --- */
int main() {
    // A. Load Data
    // Note: Ideally you use the TEST set (t10k-images...), but we can use TRAIN for a sanity check
    MNIST mnist = load_mnist(
        "data/mnist/train-images.idx3-ubyte", 
        "data/mnist/train-labels.idx1-ubyte"
    );
    
    printf("Loaded %d images for testing.\n", mnist.num);

    // B. Load Model
    GraphContext* param_ctx = init_graph();
    MLP model = init_mlp(param_ctx);
    
    if (!load_model_bin(&model, "mnist_model.bin")) {
        printf("CRITICAL ERROR: Could not load 'mnist_model.bin'. Did you train and save it?\n");
        return 1;
    }
    printf("Model weights loaded successfully!\n");

    // C. Run Inference
    int correct = 0;
    int samples_to_test = 10000; // Let's test the first 10,000 images
    if (samples_to_test > mnist.num) samples_to_test = mnist.num;

    printf("Testing on %d samples...\n", samples_to_test);

    for (int i = 0; i < samples_to_test; i++) {
        GraphContext* ctx = init_graph();

        // 1. Prepare Input (With memory fix)
        double* img_data = malloc(784 * sizeof(double));
        for (int k = 0; k < 784; k++) img_data[k] = mnist.images[i * 784 + k] / 255.0;
        Tensor* x = Param(ctx, img_data, 1, 784);
        free(img_data);

        // 2. Forward Pass
        Tensor* logits = mlp_forward(ctx, &model, x);
        int pred = argmax(logits);

        // 3. Check Accuracy
        if (pred == mnist.labels[i]) correct++;

        // 4. Cleanup
        // reset_tape NOT needed because we are freeing the whole context immediately
        free_graph(ctx);

        if ((i + 1) % 1000 == 0) printf("Processed %d... Current Acc: %.2f%%\n", i + 1, (double)correct / (i + 1) * 100.0);
    }

    printf("\n--------------------------------\n");
    printf("FINAL ACCURACY: %.2f%%\n", (double)correct / samples_to_test * 100.0);
    printf("--------------------------------\n");

    // D. Visual Sanity Check (Show me a few mistakes!)
    printf("\n--- Visual Error Analysis (Showing first 3 mistakes) ---\n");
    int mistakes_shown = 0;
    for (int i = 0; i < samples_to_test && mistakes_shown < 3; i++) {
         GraphContext* ctx = init_graph();
         double* img_data = malloc(784 * sizeof(double));
         for (int k = 0; k < 784; k++) img_data[k] = mnist.images[i * 784 + k] / 255.0;
         Tensor* x = Param(ctx, img_data, 1, 784);
         free(img_data);

         Tensor* logits = mlp_forward(ctx, &model, x);
         int pred = argmax(logits);
         int label = mnist.labels[i];

         if (pred != label) {
             mistakes_shown++;
             printf("\nMistake #%d (Label: %d, Pred: %d):\n", mistakes_shown, label, pred);
             for(int r=0; r<28; r++){
                 for(int c=0; c<28; c++){
                     printf("%c", mnist.images[i*784 + r*28+c] > 100 ? '#' : '.');
                 }
                 printf("\n");
             }
         }
         free_graph(ctx);
    }

    free_mnist(&mnist);
    free_graph(param_ctx);
    return 0;
}