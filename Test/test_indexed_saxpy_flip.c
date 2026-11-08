//------------------------------------------------------------------------------
// Regression: generic indexed SAXPY with unequal operand sizes and CSR/CSC.
// SPDX-License-Identifier: Apache-2.0
//
// Standalone public-API test (no MATLAB):
// cc -IInclude Test/test_indexed_saxpy_flip.c -Lbuild -Wl,-rpath,build \
//    -lgraphblas -o /tmp/test_indexed_saxpy_flip
//------------------------------------------------------------------------------

#include <GraphBLAS.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { uint64_t value [8] ; } Big ;

static void check (GrB_Info info)
{
    if (info == GrB_SUCCESS) return ;
    fprintf (stderr, "GraphBLAS error: %d\n", info) ;
    exit (EXIT_FAILURE) ;
}

static void multiply
(
    void *z, const void *x, GrB_Index ix, GrB_Index jx,
    const void *y, GrB_Index iy, GrB_Index jy, const void *theta
)
{
    (void) theta ;
    *(uint64_t *) z = *(const bool *) y ?
        ((const Big *) x)->value [7] + 100 * ix + 10 * jx + iy + jy : 0 ;
}

static void run (bool by_row, int sparsity, GrB_Type big, GrB_Semiring semiring)
{
    GrB_Matrix A = NULL, B = NULL, C = NULL ;
    GrB_Descriptor descriptor = NULL ;
    check (GrB_Matrix_new (&A, big, 2, 3)) ;
    check (GrB_Matrix_new (&B, GrB_BOOL, 3, 4)) ;
    check (GrB_Matrix_new (&C, GrB_UINT64, 2, 4)) ;
    int orientation = by_row ? GrB_ROWMAJOR : GrB_COLMAJOR ;
    check (GrB_set (A, orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    check (GrB_set (B, orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    check (GrB_set (C, orientation, GrB_STORAGE_ORIENTATION_HINT)) ;
    check (GxB_Matrix_Option_set_INT32 (C, GxB_SPARSITY_CONTROL, sparsity)) ;
    Big a = { { 0 } } ;
    a.value [7] = 1234 ;
    check (GrB_Matrix_setElement_UDT (A, &a, 0, 1)) ;
    a.value [7] = 5678 ;
    check (GrB_Matrix_setElement_UDT (A, &a, 1, 2)) ;
    check (GrB_Matrix_setElement_BOOL (B, true, 1, 2)) ;
    check (GrB_Matrix_setElement_BOOL (B, true, 2, 0)) ;
    check (GrB_Descriptor_new (&descriptor)) ;
    check (GrB_Descriptor_set (descriptor, GxB_AxB_METHOD, GxB_AxB_SAXPY)) ;
    check (GrB_mxm (C, NULL, NULL, semiring, A, B, descriptor)) ;
    check (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    uint64_t x = 0, y = 0 ;
    GrB_Index count = 0 ;
    check (GrB_Matrix_extractElement_UINT64 (&x, C, 0, 2)) ;
    check (GrB_Matrix_extractElement_UINT64 (&y, C, 1, 0)) ;
    check (GrB_Matrix_nvals (&count, C)) ;
    if (x != 1247 || y != 5800 || count != 2)
    {
        fprintf (stderr, "Incorrect result: row=%d format=%d x=%llu y=%llu count=%llu\n",
            by_row, sparsity, (unsigned long long) x, (unsigned long long) y,
            (unsigned long long) count) ;
        exit (EXIT_FAILURE) ;
    }
    GrB_free (&A) ; GrB_free (&B) ; GrB_free (&C) ; GrB_free (&descriptor) ;
}

int main (void)
{
    check (GrB_init (GrB_NONBLOCKING)) ;
    // Exercise the generic kernels rather than JIT-generated multiplication.
    check (GxB_Global_Option_set_INT32 (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;
    GrB_Type big = NULL ;
    GxB_IndexBinaryOp indexed = NULL ;
    GrB_BinaryOp op = NULL ;
    GrB_Scalar theta = NULL ;
    GrB_Semiring semiring = NULL ;
    check (GrB_Type_new (&big, sizeof (Big))) ;
    check (GxB_IndexBinaryOp_new (&indexed, multiply, GrB_UINT64, big, GrB_BOOL,
        GrB_UINT64, "test_indexed_saxpy_flip", NULL)) ;
    check (GrB_Scalar_new (&theta, GrB_UINT64)) ;
    check (GrB_Scalar_setElement_UINT64 (theta, 0)) ;
    check (GxB_BinaryOp_new_IndexOp (&op, indexed, theta)) ;
    check (GrB_Semiring_new (&semiring, GrB_PLUS_MONOID_UINT64, op)) ;
    for (int by_row = 0 ; by_row <= 1 ; by_row++)
    {
        run (by_row, GxB_SPARSE, big, semiring) ;
        run (by_row, GxB_BITMAP, big, semiring) ;
    }
    GrB_free (&semiring) ; GrB_free (&op) ; GrB_free (&indexed) ;
    GrB_free (&theta) ; GrB_free (&big) ;
    check (GrB_finalize ( )) ;
    puts ("Indexed SAXPY: CSR/CSC, sparse/bitmap passed") ;
    return EXIT_SUCCESS ;
}
