#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <emscripten/emscripten.h>
#include "../include/tensor.h"
#include "weights.h" 



/* --- 1. Re-define Model (Same as main.c) --- */
typedef struct {
    Tensor *W1, *b1, *W2, *b2, *W3, *b3, *W4, *b4;
} MLP;

MLP model;
GraphContext* ctx;

// Helper to copy static weight data (from weights.h) into the Tensor
void load_weight(Tensor* t, const double* src) {
    memcpy(t->storage->data, src, t->rows * t->cols * sizeof(double));
}

/* --- 2. Initialize System (Called once by JS) --- */
EMSCRIPTEN_KEEPALIVE
void init_system() {
    ctx = init_graph();
    
    // Allocate tensors
    model.W1 = Param(ctx, NULL, 784, 256);
    model.b1 = Param(ctx, NULL, 1, 256);
    model.W2 = Param(ctx, NULL, 256, 128);
    model.b2 = Param(ctx, NULL, 1, 128);
    model.W3 = Param(ctx, NULL, 128, 64);
    model.b3 = Param(ctx, NULL, 1, 64);
    model.W4 = Param(ctx, NULL, 64, 10);
    model.b4 = Param(ctx, NULL, 1, 10);

    // Load the trained data from weights.h
    load_weight(model.W1, W1_data);
    load_weight(model.b1, b1_data);
    load_weight(model.W2, W2_data);
    load_weight(model.b2, b2_data);
    load_weight(model.W3, W3_data);
    load_weight(model.b3, b3_data);
    load_weight(model.W4, W4_data);
    load_weight(model.b4, b4_data);

    printf("WASM: Model Initialized and Weights Loaded.\n");
}

/* --- 3. Prediction Function (Called by JS) --- */
// Takes a pointer to an array of 784 doubles (the image)
// Returns a pointer to an array of 10 doubles (the probabilities)
EMSCRIPTEN_KEEPALIVE
double* predict(double* img_data) {
    // 1. Create a temporary graph for this single prediction
    GraphContext* pred_ctx = init_graph();

    // 2. Wrap the input data in a Tensor
    // Note: We use Param but we copy the data from the pointer JS gave us
    Tensor* x = Param(pred_ctx, img_data, 1, 784);

    // 3. Forward Pass (Re-implementing logic here to avoid external deps)
    Tensor* h1 = Relu(pred_ctx, Add(pred_ctx, MatMul(pred_ctx, x, model.W1), model.b1));
    Tensor* h2 = Relu(pred_ctx, Add(pred_ctx, MatMul(pred_ctx, h1, model.W2), model.b2));
    Tensor* h3 = Relu(pred_ctx, Add(pred_ctx, MatMul(pred_ctx, h2, model.W3), model.b3));
    Tensor* logits = Add(pred_ctx, MatMul(pred_ctx, h3, model.W4), model.b4);

    // 4. Softmax (To get probabilities 0.0 - 1.0)
    // We compute this manually here to keep it simple
    static double probs[10];
    double max_val = -1e9;
    for(int i=0; i<10; i++) if(logits->storage->data[i] > max_val) max_val = logits->storage->data[i];
    
    double sum_exp = 0.0;
    for(int i=0; i<10; i++) sum_exp += exp(logits->storage->data[i] - max_val);
    
    for(int i=0; i<10; i++) {
        probs[i] = exp(logits->storage->data[i] - max_val) / sum_exp;
    }

    // 5. Cleanup
    free_graph(pred_ctx); 

    // Return pointer to our static probability array
    return probs;
}