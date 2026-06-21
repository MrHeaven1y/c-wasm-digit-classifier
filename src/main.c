#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/tensor.h"
#include "../include/mnist.h"

/* ---------- Data Utils ---------- */

Tensor* image_to_tensor(GraphContext* ctx, unsigned char* img) {
    double* data = malloc(784 * sizeof(double));
    for (int i = 0; i < 784; i++)
        data[i] = img[i] / 255.0;
    
    Tensor* t = Param(ctx, data, 1, 784);
    free(data);
    return t;
}

Tensor* label_to_onehot(GraphContext* ctx, unsigned char label) {
    double* y = calloc(10, sizeof(double));
    y[label] = 1.0;
    
    Tensor* t = Param(ctx, y, 1, 10);
    free(y); 
    return t;
}

/* ---------- Model ---------- */

typedef struct {
    Tensor *W1,*b1,*W2,*b2,*W3,*b3,*W4,*b4;
} MLP;

MLP init_mlp(GraphContext* ctx) {
    return (MLP){
        RandParam(ctx,784,256), Param(ctx,calloc(256,sizeof(double)),1,256),
        RandParam(ctx,256,128), Param(ctx,calloc(128,sizeof(double)),1,128),
        RandParam(ctx,128,64),  Param(ctx,calloc(64,sizeof(double)),1,64),
        RandParam(ctx,64,10),   Param(ctx,calloc(10,sizeof(double)),1,10)
    };
}

Tensor* mlp_forward(GraphContext* ctx, MLP* m, Tensor* x) {
    Tensor* h1 = Relu(ctx, Add(ctx, MatMul(ctx,x,m->W1), m->b1));
    Tensor* h2 = Relu(ctx, Add(ctx, MatMul(ctx,h1,m->W2), m->b2));
    Tensor* h3 = Relu(ctx, Add(ctx, MatMul(ctx,h2,m->W3), m->b3));
    return Add(ctx, MatMul(ctx,h3,m->W4), m->b4);
}

/* --- Save Method 1: Binary (For C Desktop App) --- */
void save_tensor_bin(FILE* f, Tensor* t) {
    // Save dimensions for safety check
    fwrite(&t->rows, sizeof(int), 1, f);
    fwrite(&t->cols, sizeof(int), 1, f);
    // Save raw double data
    fwrite(t->storage->data, sizeof(double), t->rows * t->cols, f);
}

void save_model_bin(MLP* m, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    // Save all 8 parameters in order
    save_tensor_bin(f, m->W1);
    save_tensor_bin(f, m->b1);
    save_tensor_bin(f, m->W2);
    save_tensor_bin(f, m->b2);
    save_tensor_bin(f, m->W3);
    save_tensor_bin(f, m->b3);
    save_tensor_bin(f, m->W4);
    save_tensor_bin(f, m->b4);

    fclose(f);
    printf("Binary model saved to: %s\n", filename);
}

/* --- Save Method 2: C Header (For WebAssembly/Browser) --- */
// This writes a file that you can actually compiling into your WASM program!
void save_tensor_header(FILE* f, const char* name, Tensor* t) {
    fprintf(f, "const double %s[%d] = {\n    ", name, t->rows * t->cols);
    for (int i = 0; i < t->rows * t->cols; i++) {
        fprintf(f, "%.15g,", t->storage->data[i]); // High precision
        if ((i + 1) % 10 == 0) fprintf(f, "\n    "); // Newline every 10 nums
    }
    fprintf(f, "\n};\n\n");
}

void export_model_to_header(MLP* m, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not create header file %s\n", filename);
        return;
    }

    fprintf(f, "#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n");
    
    // Write array definitions
    save_tensor_header(f, "W1_data", m->W1);
    save_tensor_header(f, "b1_data", m->b1);
    save_tensor_header(f, "W2_data", m->W2);
    save_tensor_header(f, "b2_data", m->b2);
    save_tensor_header(f, "W3_data", m->W3);
    save_tensor_header(f, "b3_data", m->b3);
    save_tensor_header(f, "W4_data", m->W4);
    save_tensor_header(f, "b4_data", m->b4);

    fprintf(f, "#endif\n");
    fclose(f);
    printf("WebAssembly Header exported to: %s\n", filename);
}

/* --- Load Helper --- */
void load_tensor_bin(FILE* f, Tensor* t) {
    int rows, cols;
    
    // Read dimensions from file
    size_t r1 = fread(&rows, sizeof(int), 1, f);
    size_t r2 = fread(&cols, sizeof(int), 1, f);

    if (r1 != 1 || r2 != 1) {
        printf("Error reading file structure.\n");
        exit(1);
    }

    // Safety Check: Do the file dimensions match the model we built?
    if (rows != t->rows || cols != t->cols) {
        printf("Shape Mismatch! File has (%d,%d) but Model expects (%d,%d)\n", 
               rows, cols, t->rows, t->cols);
        exit(1);
    }

    // Read the actual weight data directly into the tensor storage
    fread(t->storage->data, sizeof(double), rows * cols, f);
}

/* --- Main Load Function --- */
int load_model_bin(MLP* m, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("No saved model found at '%s'. Starting training from scratch.\n", filename);
        return 0; // Return 0 indicating failure/no file
    }

    printf("Found saved model! Loading weights from %s...\n", filename);

    // MUST be in the exact same order as the save function!
    load_tensor_bin(f, m->W1);
    load_tensor_bin(f, m->b1);
    load_tensor_bin(f, m->W2);
    load_tensor_bin(f, m->b2);
    load_tensor_bin(f, m->W3);
    load_tensor_bin(f, m->b3);
    load_tensor_bin(f, m->W4);
    load_tensor_bin(f, m->b4);

    fclose(f);
    printf("Model loaded successfully.\n");
    return 1; // Return 1 indicating success
}

int main() {
    srand(time(NULL));

    MNIST mnist = load_mnist(
        "data/mnist/train-images.idx3-ubyte",
        "data/mnist/train-labels.idx1-ubyte"
    );

    printf("MNIST samples: %d\n", mnist.num);
    printf("First label: %d\n", mnist.labels[0]);
    printf("First pixel: %d\n", mnist.images[0]);

    /* Persistent parameter graph */
    GraphContext* param_ctx = init_graph();
    MLP model = init_mlp(param_ctx);
    int loaded = load_model_bin(&model, "mnist_model.bin");

    // Optional: Lower learning rate if we are fine-tuning a loaded model
    double lr = loaded ? 0.001 : 0.01;
    int epochs = 3;
    double beta = 0.9;

    for (int e = 0; e < epochs; e++) {
        double epoch_loss = 0.0;

        for (int i = 0; i < mnist.num; i++) {
            /* Per-batch graph */
            GraphContext* ctx = init_graph();

            Tensor* x = image_to_tensor(ctx, &mnist.images[i * 784]);
            Tensor* y = label_to_onehot(ctx, mnist.labels[i]);

            Tensor* logits = mlp_forward(ctx, &model, x);
            Tensor* loss = Softmax_Cross_Entropy(ctx, logits, y);

            zero_grad(param_ctx);
            backward(ctx, loss);
            Optimizer_SGD_Momentum(param_ctx, lr, beta);

            epoch_loss += loss->storage->data[0];

            /* ONLY free ops + inputs */
            reset_tape(ctx);
            free(ctx);

            if (i % 1000 == 0)
                printf("Epoch %d Step %d Loss %.4f\n", e, i, loss->storage->data[0]);
        }

        printf("Epoch %d Avg Loss %.4f\n", e, epoch_loss / mnist.num);
    }


    save_model_bin(&model, "mnist_model.bin");

    // 2. Export header (The "Golden Ticket" for your website)
    export_model_to_header(&model, "weights.h");
    free_mnist(&mnist);
    free_graph(param_ctx);

    return 0;
}
