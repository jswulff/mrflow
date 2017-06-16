import numpy as np
from libcpp cimport bool

cdef extern from "image.h":
    cdef cppclass image:
        image(int width, int height, int nlabels, int truncation, int neighborhood) except +
        void solve(int *unaries, int lambd, int *labels_out, float *weights_horizontal, float *weights_vertical, float *weights_ne, float *weights_se, bool use_trws, bool effective_part_opt)


cdef class Eff_TRWS:
    cdef image *thisptr
    cdef int width
    cdef int height
    cdef int truncation

    def __cinit__(self, width, height, nlabels, truncation=1, neighborhood=4):
        self.thisptr = new image(width,height,nlabels,truncation,neighborhood)
        self.width = width
        self.height = height
        self.truncation=truncation

    def __dealloc__(self):
        del self.thisptr

    def solve(self, unaries, int lambd, labels_out, weights_horizontal=None, weights_vertical=None, weights_ne=None, weights_se=None, bool use_trws=False, bool effective_part_opt=True):
        labels_out_np = np.zeros((self.height, self.width),dtype='int32')
        cdef int [:,:,:] unaries_c = unaries.copy()
        cdef int [:,:] labels_out_c = labels_out_np

        if weights_vertical is None:
            weights_vertical = np.ones((self.height, self.width),dtype='float32')
        if weights_horizontal is None:
            weights_horizontal = np.ones((self.height, self.width),dtype='float32')
        if weights_ne is None:
            weights_ne = np.ones((self.height, self.width),dtype='float32')
        if weights_se is None:
            weights_se = np.ones((self.height, self.width),dtype='float32')

        cdef float [:,:] weights_horizontal_c = weights_horizontal
        cdef float [:,:] weights_vertical_c = weights_vertical
        cdef float [:,:] weights_ne_c = weights_ne
        cdef float [:,:] weights_se_c = weights_se

        self.thisptr.solve(&unaries_c[0,0,0],
                #<int>(1.0*lambd/self.truncation+0.5),
                lambd,
                &labels_out_c[0,0],
                &weights_horizontal_c[0,0],
                &weights_vertical_c[0,0],
                &weights_ne_c[0,0],
                &weights_se_c[0,0],
                use_trws,
                effective_part_opt)

        cdef int i,j
        for i in range(self.height):
            for j in range(self.width):
                labels_out[i,j] = labels_out_c[i,j]

