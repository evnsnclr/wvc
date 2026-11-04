// Reference (de)quantisation through ggml (E011): qref TYPE deq|quant IN OUT
// Build against a llama.cpp checkout's libggml-base:
//   clang -O2 -I LLAMA/ggml/include qref.c -L LLAMA/build/bin -lggml-base \
//         -Wl,-rpath,LLAMA/build/bin -o qref
// Q8_K has no to_float trait (it is ggml's intermediate type), so its exported reference
// functions are called directly.
#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"

void dequantize_row_q8_K(const void * x, float * y, int64_t k);
void quantize_row_q8_K_ref(const float * x, void * y, int64_t k);

static void *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb"); fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}

int main(int argc, char **argv) {
    if (argc != 5) { fprintf(stderr, "usage: qref TYPE deq|quant IN OUT\n"); return 2; }
    int type = atoi(argv[1]); long n; void *in = slurp(argv[3], &n);
    const struct ggml_type_traits *t = ggml_get_type_traits(type);
    FILE *o = fopen(argv[4], "wb");
    if (argv[2][0] == 'd') {
        long k = n / t->type_size * t->blck_size;
        float *y = malloc(k * sizeof(float));
        if (type == GGML_TYPE_Q8_K) dequantize_row_q8_K(in, y, k); else t->to_float(in, y, k);
        fwrite(y, sizeof(float), k, o);
    } else {
        long k = n / sizeof(float), bytes = k / t->blck_size * t->type_size;
        void *y = malloc(bytes);
        if (type == GGML_TYPE_Q8_K) quantize_row_q8_K_ref(in, y, k); else t->from_float_ref(in, y, k);
        fwrite(y, 1, bytes, o);
    }
    fclose(o); return 0;
}
