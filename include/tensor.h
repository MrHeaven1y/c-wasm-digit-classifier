#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>

typedef struct Storage
{
    double *data;
    int size;
    int ref_count;
} Storage;

typedef enum
{
    VAR,
    ADD,
    SUB,
    MUL,
    MATMUL,
    RELU,
    TRANSPOSE,
    SOFTMAX_CE
} OpType;

struct Tensor;

typedef void (*BackwardFunc)(struct Tensor *self);

typedef struct Tensor
{
    Storage *storage;
    int offset, rows, cols, size;

    int stride_row, stride_col;

    double *grad;
    double *velocity;

    struct Tensor *left;
    struct Tensor *right;

    OpType type;

    BackwardFunc op_backward;

    int visited;
    int is_param;

} Tensor;

typedef struct GraphContext
{
    Tensor **tape;
    Tensor **params;
    Tensor **inputs;

    int tape_cap, tape_count;
    int param_cap, param_count;
    int input_cap, input_count;

} GraphContext;

#define GET(t, i, j) (t->storage->data[t->offset + (i) * t->stride_row + (j) * t->stride_col])
#define SET(t, i, j, val) (t->storage->data[t->offset + (i) * t->stride_row + (j) * t->stride_col] = val)

// --- Function Prototypes ---

GraphContext *init_graph();
void reset_tape(GraphContext *ctx);
void free_graph(GraphContext *ctx);
void free_tensor(Tensor *t);

Tensor *Param(GraphContext *ctx, double *values, int rows, int cols);
Tensor *RandParam(GraphContext *ctx, int row, int col);

// Helpers (Exposed for main.c)
Storage *_alloc_storage(int size);
Tensor *_alloc_node(GraphContext *ctx, int rows, int cols, OpType type, Tensor *left, Tensor *right, int is_param);
void _track_op(GraphContext *ctx, Tensor *t);
void _track_param(GraphContext *ctx, Tensor *t);
void _track_input(GraphContext* ctx, Tensor* t);

Tensor *Add(GraphContext *ctx, Tensor *a, Tensor *b);
Tensor *Sub(GraphContext *ctx, Tensor *a, Tensor *b);
Tensor *Relu(GraphContext *ctx, Tensor *a);
Tensor *Square(GraphContext *ctx, Tensor *a);
Tensor *T(GraphContext *ctx, Tensor *a);
Tensor *MatMul(GraphContext *ctx, Tensor *a, Tensor *b);

// --- NEW: Add this missing prototype ---
Tensor *Softmax_Cross_Entropy(GraphContext *ctx, Tensor *logits, Tensor *targets);

// Engine
void backward(GraphContext *ctx, Tensor *root);
void zero_grad(GraphContext *ctx);
void Optimizer_SGD_Momentum(GraphContext *ctx, double lr, double beta);

#endif