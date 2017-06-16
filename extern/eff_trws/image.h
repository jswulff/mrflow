#ifndef WRAPPER_H
#define WRAPPER_H

#include <cstdio>
#include <ctime>
#include <cstring>
#include <cmath>
#include <assert.h>
#include "graph.h"
#include "kovtun.h"
#include "aexpand.h"
#include "TRWBP.h"

/**** Assumes a grid (image-like) graph ****/
class image
{
	public:
        image(int width, int height, int nlabel, int truncation=1, int neighborhood=4);
        virtual ~image();

        void solve(int *unaries,
                int lambd,
                int *labels_out,
                float *weights_horizontal,
                float *weights_vertical,
                float *weights_ne,
                float *weights_se,
                bool use_trws=false,
                bool effective_part_opt=true);

    private:
        int width;           // Grid width
        int height;          // Grid height
        int num_nodes;       // Number of nodes in the grid
        int num_labels;      // Number of labels
        int neighborhood;    // 4 or 8 connected NH

        Energy *eng;               // Stores the unary and pairwise costs
        unsigned char **label_map; // Label assignments to pixels


	/**** multi-label solver functions ****/
	void kovtun(bool do_aexpand=false);   // Computes the partially optimal solution using Kovtun's method.
	                                      // The remaining nodes can be solved using alpha expansion if do_expand = true.
	void trw(bool use_projection=false);  // use_projection = true : Partially optimal solution computation + TRW/BP on remaining nodes
	                                      // use_projection = false: Standard TRW/BP
	/*************************************/

};

#endif
