#include "../include/tensor.h"
#include <math.h>
#include <string.h>
#include <assert.h>

GraphContext* init_graph(){
    GraphContext* ctx = (GraphContext*) malloc(sizeof(GraphContext));

    ctx->tape_cap = 1024;
    ctx->tape = (Tensor**)malloc(sizeof(Tensor*) * ctx->tape_cap);
    ctx->tape_count = 0;

    ctx->param_cap = 64;
    ctx->params = (Tensor**)malloc(sizeof(Tensor*) * ctx->param_cap); 
    ctx->param_count = 0;

    ctx->input_cap = 64;
    ctx->inputs = (Tensor**)malloc(sizeof(Tensor*) * ctx->input_cap);   
    ctx->input_count = 0;

    return ctx;
}

Storage* _alloc_storage(int size){
    Storage* s = (Storage*)malloc(sizeof(Storage));
    s->data = (double*)calloc(size, sizeof(double));
    s->size = size;
    s->ref_count = 1;
    return s;
} 


void _track_op(GraphContext* ctx, Tensor* t){
    if(ctx->tape_count >= ctx->tape_cap){
        ctx->tape_cap *= 2;
        ctx->tape = (Tensor**)realloc(ctx->tape, sizeof(Tensor*) * ctx->tape_cap);
    }
    ctx->tape[ctx->tape_count++] = t;
}

void _track_param(GraphContext* ctx, Tensor* t){
    if(ctx->param_count >= ctx->param_cap){
        ctx->param_cap *= 2;
        ctx->params = (Tensor**)realloc(ctx->params, sizeof(Tensor*) * ctx->param_cap);
    }
    
    ctx->params[ctx->param_count++] = t;
}

void _track_input(GraphContext* ctx, Tensor* t){
    if(ctx->input_count >= ctx->input_cap){
        ctx->input_cap *= 2;
        ctx->inputs = (Tensor**)realloc(ctx->inputs, sizeof(Tensor*) * ctx->input_cap);
    }
    
    ctx->inputs[ctx->input_count++] = t;
}

Tensor* _alloc_node(GraphContext* ctx, int rows, int cols, OpType type, Tensor* left, Tensor* right, int is_param){
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    
    t->rows = rows;
    t->cols = cols;
    
    t->size = rows * cols;
    t->stride_row = cols;
    t->stride_col = 1;
    t->offset = 0;

    t->storage = NULL;
    t->grad = (double*)calloc(t->size, sizeof(double));
    t->velocity = NULL;
    t->type = type;
    t->left = left;
    t->right = right; 

    t->op_backward = NULL;
    t->visited = 0;

    if(is_param){
        _track_param(ctx, t);
    } else if(type != VAR){
        _track_op(ctx, t);
    } else{
        _track_input(ctx, t);
    }

    return t;
}

void free_tensor(Tensor* t){
    if(!t) return;

    if(t->storage){
        t->storage->ref_count--;
        if(t->storage->ref_count <= 0){
            free(t->storage->data);
            free(t->storage);
        }
    }
    
    if(t->grad) free(t->grad);
    
    // --- NEW: Free velocity if it exists ---
    if(t->velocity) free(t->velocity); 

    free(t);
}
void reset_tape(GraphContext* ctx){
    for(int i=0; i<ctx->tape_count; i++){
        free_tensor(ctx->tape[i]);
    }
    ctx->tape_count = 0;
}

void free_graph(GraphContext* ctx){
    reset_tape(ctx);
    
    for(int i=0; i<ctx->param_count; i++) free_tensor(ctx->params[i]);
    free(ctx->params);
    
    for(int i=0; i<ctx->input_count; i++) free_tensor(ctx->inputs[i]);
    free(ctx->inputs);

    free(ctx->tape);
    free(ctx);
}

Tensor* Param(GraphContext* ctx, double* values, int rows, int cols){
    
    Tensor* t = _alloc_node(ctx, rows, cols, VAR, NULL, NULL,1);
    t->storage = _alloc_storage(t->size);
    
    if(values) memcpy(t->storage->data, values, t->size * sizeof(double));
    
    return t;
}

Tensor* RandParam(GraphContext* ctx, int rows, int cols){
    Tensor* t = _alloc_node(ctx, rows, cols, VAR, NULL, NULL, 1);
    t->storage = _alloc_storage(rows * cols);

    double scale = sqrt(2.0 / rows);
    for(int i=0; i<t->size; i++){
        t->storage->data[i] = ((double)rand()/RAND_MAX * 2.0 - 1.0) * scale;
    }

    return t;
}

// --- Backward Functions ---

static void backward_add(Tensor* t){
    for(int i=0; i<t->size; i++){
        double g = t->grad[i];
        if(t->left->grad) t->left->grad[i] += g;
        if(t->right->grad) t->right->grad[i] += g;
    }
}

static void backward_sub(Tensor* t) {
    for (int i = 0; i < t->size; i++) {
        double g = t->grad[i];
        if (t->left->grad)  t->left->grad[i]  += g;
        if (t->right->grad) t->right->grad[i] -= g;
    }
}

static void backward_square(Tensor* t) {
    Tensor* input = t->left;
    for (int r = 0; r < t->rows; r++) {
        for (int c = 0; c < t->cols; c++) {
            double x = GET(input, r, c);
            int idx = r * t->cols + c;
            if (input->grad) {
                input->grad[idx] += t->grad[idx] * 2.0 * x;
            }
        }
    }
}

static void backward_relu(Tensor* t){
    for(int r=0; r < t->rows; r++){
        for(int c=0; c< t->cols; c++){
            double val = GET(t->left, r, c);
            double mask = (val > 0.0 ? 1.0 : 0.0); // Fixed derivative (1.0 or 0.0)
            
            int idx = r * t->cols + c;
            if(t->left->grad) t->left->grad[idx] += mask * t->grad[idx];
        }
    }
}

void backward_softmax_ce(Tensor* node) {
    // FIX 1: Use left/right instead of parents
    Tensor* logits = node->left;
    Tensor* targets = node->right;
    
    int batch_size = logits->rows;
    int num_classes = logits->cols;
    double* z = logits->storage->data;
    double* y = targets->storage->data;
    double* dz = logits->grad; 
    
    for (int b = 0; b < batch_size; b++) {
        // ... (Max finding logic remains the same) ...
        double max_val = -1e9;
        for (int c = 0; c < num_classes; c++) {
            if (z[b * num_classes + c] > max_val) max_val = z[b * num_classes + c];
        }
        
        // ... (Sum Exp logic remains the same) ...
        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += exp(z[b * num_classes + c] - max_val);
        }

        for (int c = 0; c < num_classes; c++) {
            double prob = exp(z[b * num_classes + c] - max_val) / sum_exp;
            double target = y[b * num_classes + c];
            
            // dL/dz = (p - y) / batch_size * incoming_grad
            if(node->grad) {
                dz[b * num_classes + c] += (prob - target) / batch_size * node->grad[0];
            }
        }
    }
}

static void backward_transpose(Tensor* t){
    Tensor* input = t->left; 
    for(int r=0; r< t->rows; r++){
        for(int c=0; c< t->cols; c++){
            double g = t->grad[r * t->cols + c];
            int inp_idx = r * input->cols + c; // Logic: input is (cols, rows) relative to t
            // But wait, actually simpler: 
            // t[r, c] comes from input[c, r]. 
            // So we add t->grad[r,c] to input->grad[c,r].
            int correct_inp_idx = c * input->cols + r; // Fixed index logic
            
            if(input->grad) input->grad[correct_inp_idx] += g;
        }
    }
}

static void backward_matmul(Tensor* t){
    Tensor* A = t->left;
    Tensor* B = t->right;
    
    if(A->grad){
        for(int m=0; m < A->rows; m++){
            for(int n=0; n < A->cols; n++){
                double sum = 0.0;
                for(int k=0; k < B->cols; k++){
                    sum += t->grad[m * t->cols + k] * GET(B, n, k); 
                }
                A->grad[m * A->cols + n] += sum;
            }
        }
    }
    
    if(B->grad){
        for(int n=0; n < B->rows; n++){
            for(int k=0; k < B->cols; k++){
                double sum = 0.0;
                for(int m=0; m < A->rows; m++){
                    sum += GET(A, m, n) * t->grad[m * t->cols + k];
                }
                B->grad[n * B->cols + k] += sum;
            }
        }
    }
}

// --- Constructors ---

Tensor* Softmax_Cross_Entropy(GraphContext* ctx, Tensor* logits, Tensor* targets) {
    // 1. Check Dimensions
    int batch_size = logits->rows;
    int num_classes = logits->cols;
    
    // 2. Alloc Output Node (Scalar Loss)
    // FIX: Use SOFTMAX_CE enum directly
    Tensor* out = _alloc_node(ctx, 1, 1, SOFTMAX_CE, logits, targets, 0); 
    out->storage = _alloc_storage(1);

    // 3. Forward Pass Logic
    double total_loss = 0.0;
    double* z = logits->storage->data;
    double* y = targets->storage->data;

    for (int b = 0; b < batch_size; b++) {
        // --- A. Find Max (for numerical stability) ---
        double max_val = -1e9;
        for (int c = 0; c < num_classes; c++) {
            double val = z[b * num_classes + c];
            if (val > max_val) max_val = val;
        }

        // --- B. Compute Sum of Exponentials ---
        double sum_exp = 0.0;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += exp(z[b * num_classes + c] - max_val);
        }

        // --- C. Compute Log-Sum-Exp ---
        double log_sum_exp = max_val + log(sum_exp);

        // --- D. Cross Entropy Loss ---
        // Loss = - Sum(y_i * log(p_i))
        // log(p_i) = z_i - log_sum_exp
        for (int c = 0; c < num_classes; c++) {
            if (y[b * num_classes + c] > 0.0) { // If this is the target class
                total_loss += (log_sum_exp - z[b * num_classes + c]);
            }
        }
    }

    // Average loss over batch
    out->storage->data[0] = total_loss / batch_size;

    // FIX: Ensure backward pointer is set
    out->op_backward = backward_softmax_ce;

    return out;
}

Tensor* Square(GraphContext* ctx, Tensor* a) {
    Tensor* t = _alloc_node(ctx, a->rows, a->cols, VAR, a, NULL, 0);   
    t->storage = _alloc_storage(t->size);

    for (int r = 0; r < t->rows; r++) {
        for (int c = 0; c < t->cols; c++) {
            double val = GET(a, r, c);
            SET(t, r, c, val * val);
        }
    }

    t->op_backward = backward_square;
    return t;
}

Tensor* Add(GraphContext* ctx, Tensor* a, Tensor* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Tensor* t = _alloc_node(ctx, a->rows, a->cols, ADD, a, b, 0);
    t->storage = _alloc_storage(t->size);

    for (int r = 0; r < t->rows; r++) {
        for (int c = 0; c < t->cols; c++) {
            double val = GET(a, r, c) + GET(b, r, c);
            SET(t, r, c, val);
        }
    }
    
    t->op_backward = backward_add;
    return t;
}

Tensor* Sub(GraphContext* ctx, Tensor* a, Tensor* b) {
    assert(a->rows == b->rows && a->cols == b->cols);
    Tensor* t = _alloc_node(ctx, a->rows, a->cols, SUB, a, b, 0);
    t->storage = _alloc_storage(t->size);

    for (int r = 0; r < t->rows; r++) {
        for (int c = 0; c < t->cols; c++) {
            double val = GET(a, r, c) - GET(b, r, c);
            SET(t, r, c, val);
        }
    }
    
    t->op_backward = backward_sub;
    return t;
}

Tensor* Relu(GraphContext* ctx, Tensor* a) {
    Tensor* t = _alloc_node(ctx, a->rows, a->cols, RELU, a, NULL, 0);
    t->storage = _alloc_storage(t->size);

    for (int i = 0; i < t->rows; i++) {
        for (int j = 0; j < t->cols; j++) {
            double val = GET(a, i, j);
            SET(t, i, j, (val > 0.0 ? val : 0.0));
        }
    }

    t->op_backward = backward_relu;
    return t;
}

Tensor* MatMul(GraphContext* ctx, Tensor* a, Tensor* b){
    if(a->cols != b->rows){
        fprintf(stderr, "MatMul Shape Error: (%d, %d) x (%d, %d)\n", 
                a->rows, a->cols, b->rows, b->cols);
        exit(1);
    }

    Tensor* t = _alloc_node(ctx, a->rows, b->cols, MATMUL, a, b, 0);
    t->storage = _alloc_storage(t->size);

    for(int i=0; i < t->rows; i++){
        for(int k=0; k < t->cols; k++){
            double sum = 0.0;
            for(int j=0; j < a->cols; j++){
                sum += GET(a,i,j) * GET(b,j,k);
            }
            SET(t, i, k, sum);
        }
    }

    t->op_backward = backward_matmul;
    return t;
}

Tensor* T(GraphContext* ctx, Tensor* a){
    Tensor* t = _alloc_node(ctx, a->cols, a->rows, TRANSPOSE, a, NULL, 0); 

    t->storage = a->storage;
    t->storage->ref_count++;

    t->stride_row = a->stride_col;
    t->stride_col = a->stride_row;

    t->offset = a->offset;
    t->op_backward = backward_transpose;

    return t;
}

// --- Engine ---

// Pass Tensor*** sorted to allow reallocating the array pointer
static void _build_topo(Tensor* v, Tensor*** sorted, int* idx, int* cap) {
    if (v->visited) return;
    v->visited = 1;

    if (v->left)  _build_topo(v->left, sorted, idx, cap);
    if (v->right) _build_topo(v->right, sorted, idx, cap);

    // Check capacity and grow if necessary
    if (*idx >= *cap) {
        *cap *= 2;
        *sorted = (Tensor**)realloc(*sorted, sizeof(Tensor*) * (*cap));
    }

    (*sorted)[(*idx)++] = v;
}

void backward(GraphContext* ctx, Tensor* root){
    // 1. Reset visited flags for known tracked lists
    // Note: X and Y are not here, so their flags are currently dirty (1) or clean (0)
    for (int i = 0; i < ctx->tape_count; i++) ctx->tape[i]->visited = 0;
    for (int i = 0; i < ctx->param_count; i++) ctx->params[i]->visited = 0;
    for (int i = 0; i < ctx->input_count; i++) ctx->inputs[i]->visited = 0;
    
    // 2. Dynamic Allocation Setup
    // Start with a safe guess, but allow growth for inputs/targets (X, Y)
    int max_nodes = ctx->tape_count + ctx->param_count + ctx->input_count; 
    Tensor** sorted = (Tensor**)malloc(sizeof(Tensor*) * max_nodes);
    int idx = 0;

    // 3. Build Topo Sort (Pass address of 'sorted' to allow realloc)
    _build_topo(root, &sorted, &idx,&max_nodes);

    // 4. Initialize Loss Grad
    if(root->grad) for(int i=0; i<root->size; i++) root->grad[i] = 1.0;
    // 5. Reverse Pass
    for (int i = idx - 1; i >= 0; i--) {
        Tensor* t = sorted[i];
        if (t->op_backward) t->op_backward(t);
        
    }

    // 6. CRITICAL CLEANUP: Reset visited flags for ALL nodes in the graph
    // This catches X and Y, ensuring they are ready for the next epoch.
    for(int i=0; i<idx; i++) sorted[i]->visited = 0;

    free(sorted);
}

void Optimizer_SGD_Momentum(GraphContext* ctx, double lr, double beta){
    // Iterate over all learnable parameters
    for(int i = 0; i < ctx->param_count; i++){
        Tensor* p = ctx->params[i];
        
        // 1. Lazy Allocation: Create velocity buffer if it doesn't exist yet
        if (!p->velocity) {
            p->velocity = (double*)calloc(p->size, sizeof(double));
        }

        // 2. Apply Momentum Update
        // Note: Params are always contiguous, so we can iterate 0..size
        for(int j = 0; j < p->size; j++){
            // Your formula: v = beta*v + (1-beta)*grad
            p->velocity[j] = beta * p->velocity[j] + (1.0 - beta) * p->grad[j];
            
            // Update Weight: w = w - lr * v
            // FIX: Access storage->data, not w->data
            p->storage->data[j] -= lr * p->velocity[j]; 
        }
    }
}

void zero_grad(GraphContext* ctx) {
    for (int i = 0; i < ctx->param_count; i++) {
        Tensor* p = ctx->params[i];
        if (p->grad) {
            memset(p->grad, 0, p->size * sizeof(double));
        }
    }
}