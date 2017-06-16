#include <cmath>
#include <ctime>
#include <cstdio>
#include "graph.h"
#include "energy.h"
#include "includes.h"

typedef Graph<long, long, long> Grapht;

// Computes the partially optimal solution using Kovtun's method
//
// More details can be found in:
//      I. Kovtun, "Partial optimal labeling search for a NP-Hard subclass of (max,+) problems"
//      DAGM Symposium, pp. 402-409, 2003.

class Kovtun
{
	// No. of unary, pairwise terms, and labels respectively
	int nvar;
	int npair;
	int nlabel;

	void initialize()
	{
		nvar  = energy->nvar;
		npair = energy->npair;
		nlabel= energy->nlabel;
		multiSolution = new unsigned char [nvar];
		labelCount = new int[nlabel];
		multiSolutionCount = 0;

		if(ismult) nodes = new Grapht::node_id[nlabel*nvar];
		else nodes = new Grapht::node_id[nvar];

		for (int i=0;i<nvar;i++)
			multiSolution[i] = NOLABEL;
		for (int i=0;i<nlabel;i++)
			labelCount[i]=0;
	}

	int minUcost(int var, int label)
	{
		int mincost = INFINITE_D;
		for (int i=0;i<nlabel;i++)
			if ((i!=label)&&(mincost > energy->unaryCost[var][i])) 
				mincost = energy->unaryCost[var][i];

		return mincost;
	}

public:
	Energy* energy;			// The corresponding energy,
	Grapht** graph;			// and the graph.
	unsigned char *multiSolution;	// multiSolution[x_i] contains label for node x_i.
	int *labelCount;		// labelCount[i] contains no. of nodes labelled 'i'.
	int multiSolutionCount;		// Sum_i(labelCount[i])
	int ismult;
	Grapht::node_id* nodes;

	Kovtun(Energy *e, Grapht** g, int ismultiple)
	{ // Uses multiple graphs
		energy= e;
		ismult= ismultiple;
		graph = g;
		initialize();
	}

	Kovtun(Energy *e, Grapht* g, int ismultiple)
	{ // Uses a single graph
		energy= e;
		ismult= ismultiple;
		graph = new Grapht* [1];
		graph[0] = g;
		initialize();
	}

	~Kovtun()
	{
		delete [] multiSolution;
		delete [] labelCount;
		delete [] nodes;
	}

	void minimize()
	{
		if(ismult)
			for (int i=0;i<nlabel;i++)
			{
				findPersistent(i);
				//graph[i]->reset();
			}
		else
			for (int i=0;i<nlabel;i++)
			{
				findPersistent(i);
				graph[0]->reset();
			}
	}

	// Solve the auxilliary problem for label eLabel
	double findPersistent(unsigned char eLabel)
	{
		int i, j;
		int countActiveU, countActiveP;
		int offset = 0, goffset = 0;
		if(ismult)
		{
			offset = nvar*eLabel;
			goffset= eLabel;
		}

		countActiveU = energy->countActiveU;
		countActiveP = energy->countActiveP;

		// Add Nodes, Unary Costs
		for (j=0; j<countActiveU; j++)
		{
			i = energy->activeU[j];
			nodes[offset+i] = graph[goffset]->add_node();
			graph[goffset]->edit_tweights(nodes[offset+i], energy->unaryCost[i][eLabel], minUcost(i,eLabel));
		}

		// Add pairwise costs 
		int from, to, weight;
		for (j=0; j<countActiveP; j++)
		{
			i = energy->activeP[j];
			from = energy->pairIndex[i][0];	
			to = energy->pairIndex[i][1];
			weight = energy->pairCost[i];
			graph[goffset]->add_edge(nodes[offset+from], nodes[offset+to], weight, weight);
		}

		clock_t start = clock();
		graph[goffset]->maxflow();
		double comptime = (double(clock())-start);

		for (j=0; j<countActiveU; j++)
		{
			i = energy->activeU[j];
			if (graph[goffset]->what_segment(nodes[offset+i])==1)
			{
				if (multiSolution[i]==NOLABEL)
				{
					multiSolutionCount += 1;
					multiSolution[i] = eLabel;
					labelCount[eLabel] += 1;
				}
			}
		}
		printf("%d: %d\n", eLabel, labelCount[eLabel]);
		return comptime;
	}

	void printSolution()
	{
		printf("\nKovtun Persistent Labels: ");
		for (int i=0;i<nvar;i++)
			printf(" %u ", multiSolution[i]);
	}
};
