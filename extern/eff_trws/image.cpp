#include "image.h"

#define OBJ 0
#define BKG 1
#define ACTIVE true
#define INACTIVE false

/************************************************************************/
/***************** See image.h for function description *****************/
/************************************************************************/

image::image(int width, int height, int nlabels, int truncation, int neighborhood)
{
    setbuf(stdout, NULL);
    printf("\tInitializing...");
	int i, j;

    this->width = width;
    this->height = height;
    this->neighborhood = neighborhood;

	num_labels = nlabels;
	num_nodes = width*height;

	label_map = new unsigned char* [height];
	for(i=0; i<height; i++)
		label_map[i] = new unsigned char [width];

    printf("Truncation = %d ", truncation);

	eng = new Energy(num_labels, truncation);
	eng->nvar = num_nodes;

    if (neighborhood == 8)
        eng->npair= 4*width*height - 3*(width+height) + 2; // Using an 8-neighbourhood
    else
        // Assume 4-connected neighborhood
        eng->npair = 2*width*height - (width+height);

	eng->allocateMemory();
    printf("done.\n");

	/**
	 ** Assign the unary (eng->unaryCost) and pairwise costs (eng->pairCost).
	 ** Note that pairwise costs are indexed by eng->pairIndex, which is also to be initialized.
	 **/
}

image::~image()
{
    printf("\nDeleting image\n");
	int i;

	for(i=0; i<height; i++)
	{
		delete[] label_map[i];
	}
	delete[] label_map;
	delete eng;
    printf("done.\n");
}

void image::solve(int *unaries,
        int lambda,
        int *labels_out,
        float *weights_horizontal,
        float *weights_vertical,
        float *weights_ne,
        float *weights_se,
        bool use_trws,
        bool effective_part_opt)
{
    // Set up energies
    int i,j,k;
    for (i=0; i < height; i++)
        for (j=0; j < width; j++)
            for (k=0; k < num_labels; k++)
            {
                //printf("%d ", (i*width+j)*num_labels+k);
                eng->unaryCost[i*width+j][k] = unaries[(i*width+j)*num_labels+k];
            }
    int pair_id = 0;
    // Horizontal connections
    for(i=0; i<height; i++)
        for(j=0; j<width-1; j++)
        {
            k = i*width + j;
            eng->pairCost[pair_id] = (int)(lambda * weights_horizontal[k]);
            eng->pairIndex[pair_id][0] = k;
            eng->pairIndex[pair_id][1] = k+1;
            pair_id++;
        }
    // Vertical connections
    for(i=0; i<height-1; i++)
        for(j=0; j<width; j++)
        {
            k = i*width + j;
            eng->pairCost[pair_id] = (int)(lambda * weights_vertical[k]);
            eng->pairIndex[pair_id][0] = k;
            eng->pairIndex[pair_id][1] = k+width;
            pair_id++;
        }

    if (this->neighborhood == 8)
    {
        // Add SE connections
        for (i=0; i<height-1; i++)
            for (j=0; j<width-1; j++)
            {
                k = i*width+j;
                eng->pairCost[pair_id] = (int)(lambda * weights_se[k]);
                eng->pairIndex[pair_id][0] = k;
                eng->pairIndex[pair_id][1] = k + width + 1;
                pair_id++;
            }
        // Add NE connections
        for (i=1; i<height; i++)
            for (j=0; j<width-1; j++)
            {
                k = i*width+j;
                eng->pairCost[pair_id] = (int)(lambda * weights_ne[k]);
                eng->pairIndex[pair_id][0] = k;
                eng->pairIndex[pair_id][1] = k - width + 1;
                pair_id++;
            }
    }




    // Compute partially optimal solution + TRW/BP
    //this->kovtun(false);

    if (use_trws)
        this->trw(effective_part_opt);
    else
        this->kovtun(effective_part_opt);

    // Copy out labels
    for (i=0; i<height; i++)
        for (j=0; j < width; j++)
            labels_out[i*width+j] = label_map[i][j];

}


            


void image::kovtun(bool do_aexpand)
{
	int i, j;
	unsigned char lbl;
	double time_np;
	Grapht** graph;

	for(i=0; i<height; i++)
	for(j=0; j<width; j++)
		label_map[i][j] = NOLABEL;

	graph = new Grapht* [num_labels];
	for(i=0; i<num_labels; i++) graph[i] = new Grapht(num_nodes, eng->npair);
	Kovtun* solver = new Kovtun(eng, graph, 1); // 1 => use multiple graphs and the Reuse method (see Alahari et al., 2008)

    printf("Projecting energy function...\n");

	/* Projecting the energy function (Reduce method, see Alahari et al., 2008) */
	for(i=0; i<num_labels; i++)
	{
        printf("num_label = %d. Finding persistent.", i);
		time_np = solver->findPersistent(i);
        printf("projecting.\n");
		eng->Project(solver->multiSolution);
	}

	if(do_aexpand)
	{
	/* Solve the remaining nodes using the alpha expansion method */
		Aexpand *solvera;
		eng->Project(solver->multiSolution);
		solvera = new Aexpand(eng, graph);
		time_np = solvera->minimize(solver->nodes);

		int count=0;
		for(i=0; i<height; i++)
		for(j=0; j<width; j++)
		{
			lbl = solver->multiSolution[i*width+j];
			if(lbl != NOLABEL)
				label_map[i][j] = lbl;
			else
			{
				label_map[i][j] = solvera->label_map[eng->activeU[count]];
				count++;
			}
		}
		delete solvera;
	}
	else
	{
        printf("Computing partially optimal solution...\n");
	/* Or compute only the partially optimal solution */
		for(i=0; i<height; i++)
		for(j=0; j<width; j++)
		{
			lbl = solver->multiSolution[i*width+j];
			if(lbl != NOLABEL)
				label_map[i][j] = lbl;
		}
	}

    printf("Cleaning up...\n");

	for(i=0; i<num_labels; i++)
		delete graph[i];
	delete [] graph;
	delete solver;
}

void image::trw(bool use_projection)
{
    printf("Initializing.");
	int i, j;
	unsigned char lbl;
	Energy* n_eng;

	for(i=0; i<height; i++)
	for(j=0; j<width; j++)
		label_map[i][j] = NOLABEL;


	if(use_projection)
	{
        printf("Constructing graphs.");
	/* Compute the partially optimal solution and solve the remaining nodes using TRW and BP */
		Grapht* graph  = new Grapht(num_nodes, eng->npair);
		Kovtun* solver = new Kovtun(eng, graph, 0); // 0 => use a single graph
		double time_np;

        printf("Projecting.");

		for(i=0; i<num_labels; i++)
		{
            printf("\n\tNumlabel = %d. step 1", i);
			time_np = solver->findPersistent(i);
            printf("\n\tNumlabel = %d. step 2", i);
			eng->Project(solver->multiSolution);
            printf("\n\tNumlabel = %d. step 3", i);
			graph->reset();
		}
        printf("\nProjecting.");
		n_eng = eng->Projection(solver->multiSolution);
        printf("Minimizing.");

		TRWBP *solvert = new TRWBP(n_eng);
		solvert->minimize(true, 70, 70); // 70 iterations each of BP and TRW

        printf("Saving solution.");

		int count=0;
		for(i=0; i<height; i++)
		for(j=0; j<width; j++)
		{
			lbl = solver->multiSolution[i*width+j];
			if(lbl != NOLABEL)
				label_map[i][j] = lbl;
			else
			{
				// solvert->solutionT contains TRW result and 
				// solvert->solutionB contains BP result
				label_map[i][j] = solvert->solutionT[count];
				solver->multiSolution[i*width+j] = label_map[i][j]; // To compute energy
				count++;
			}
		}
        printf("Cleaning up.");
		delete graph;
		delete solver;
		delete solvert;
		delete n_eng;
        printf("Quitting.");
	}
	else
	{
	/* Or solve using standard TRW and BP only */
		n_eng = eng;
		TRWBP *solver = new TRWBP(n_eng);
		solver->minimize(true, 70, 70); // 70 iterations each of BP and TRW

		// solvert->solutionT contains TRW result and 
		// solvert->solutionB contains BP result
		for(i=0; i<height; i++)
		for(j=0; j<width; j++)
			label_map[i][j] = solver->solutionT[i*width+j];
		delete solver;
	}
}


