#include <cmath>
#include <ctime>
#include <cstdio>
#include "graph.h"
#include "energy.h"
#include "includes.h"

typedef Graph<long, long, long> Grapht;

// Given a graph and its corresponding energy, computes the labelling
// using the alpha expansion algorithm.
//
// This alpha expansion implementation is based on the code available at:
// http://www.adastral.ucl.ac.uk/~vladkolm/software/match-v3.3.src.tar.gz

class Aexpand
{
	int nvar;                // No. of nodes
	int npair;               // No. of pairwise relationships
	int nlabel;              // No. of labels
	Grapht** g;
	Grapht::node_id* nodes;  // Node variables
	int max_iter;            // No. of maximum iterations
	bool rand_iter;          // Change the order of labels in each iteration
	bool *is_active;         // Stores the 'active' set of nodes
	captype truncation;
	flowtype E;              // Energy value
	flowtype constE;         // Constant energy value
	int countActiveU, countActiveP;
	bool reuse;
	
	void allocateMemory()
	{
		label_map = new unsigned char[nvar];
		is_active = new bool[nlabel*nvar];
	}

	void initialize()
	{
		nvar = energy->nvar;
		npair = energy->npair;
		countActiveU = energy->countActiveU;
		countActiveP = energy->countActiveP;
		nlabel = energy->nlabel;
		truncation = energy->truncation;
		allocateMemory();
		E = 0;
		for(int i=0; i<nvar; i++)
		{
			label_map[i] = min_unary(i);
		}
	}

public:
	Energy* energy;			// Stores the energy corresponding to the graph
	unsigned char* label_map;

	int min_unary(int var)
	{
		int minu = 0;
		for (int i=0;i<nlabel;i++)
			if (energy->unaryCost[var][minu] > energy->unaryCost[var][i])
				minu = i;
		return minu;
	}

	// (rand_iter=false) Uses the same label order for expansion moves by default
	Aexpand(Energy *e, int m_iter=10, bool r_iter=false)
	{
		max_iter = m_iter;
		rand_iter= r_iter;
		energy = e;
		reuse = false;
		initialize();
	}

	Aexpand(Energy *e, Grapht **graph, int m_iter=15, bool r_iter=false)
	{
		max_iter = m_iter;
		rand_iter= r_iter;
		energy = e;
		reuse = true;
		g = graph;
		initialize();
	}

	~Aexpand()
	{
		delete [] is_active;
		delete [] label_map;
	}

	// This is based on the alpha expansion code available at:
	// http://www.adastral.ucl.ac.uk/~vladkolm/software/match-v3.3.src.tar.gz
	double minimize(Grapht::node_id* _nodes=NULL)
	{
		unsigned char* permutation;	/* contains random permutation of 0, 1, ..., label_num-1 */
		bool* label_buf;	/* if label_buf[l] is true then expansion of label corresponding to l
					cannot decrease the energy */
		int label_buf_num;      /* number of 'false' entries in label_buf */
		int i, index, label;
		int step, iter;
		flowtype E_old;
		double comptime = 0;

		if(reuse) nodes = _nodes;

        // Fix the seed to make results repeatable
		unsigned int seed = 100; //time(NULL);
		srand(seed);

		permutation = new unsigned char[nlabel];	// change the order of expansion moves
		label_buf = new bool[nlabel];

		compute_energy();
		printf("E = %ld\n\n", E);

		for (i=0; i<nlabel; i++) label_buf[i] = false;
		label_buf_num = nlabel;
		step = 0;

		for (iter=0; iter<max_iter && label_buf_num>0; iter++)
		{
			if (iter==0 || rand_iter)
				generate_permutation(permutation, nlabel);
        
			for (index=0; index<nlabel; index++)
			{
				label = permutation[index];
				if (label_buf[label]) continue;
        
				E_old = E;
				double expand_time = expand(label);
				comptime += expand_time;
				step++;
        
				if (E_old == E)
				{
					printf("-");
					if (!label_buf[label]) { label_buf[label] = true; label_buf_num--; }
				}
				else
				{
					printf("*");
					for (i=0; i<nlabel; i++) label_buf[i] = false;
					label_buf[label] = true;
					label_buf_num = nlabel - 1;
				}
			}
			printf(" E = %ld, t = %lf\n", E, comptime/CLOCKS_PER_SEC); fflush(stdout);
		}
		printf("%.1f iterations\n", ((float)step)/nlabel);

		delete [] permutation;
		delete [] label_buf;
		return comptime;
	}

	double expand(unsigned char label)
	{ // Assuming graphs are recycled
		int i, j, offset;
		unsigned char x1[3], x2[3];
		captype Ra, Ra_bar;
		Grapht::node_id a;
		flowtype E_old;
		unsigned char label_bar;
		unsigned char* tmplabel_map;
		double comptime = 0;
		captype *tmp_tweight;

		tmplabel_map = new unsigned char [nvar];
		tmp_tweight = new captype [nvar];
		for(i=0; i<nvar; i++) tmp_tweight[i] = 0;

		constE  = 0;
		comptime= 0;
		offset  = label*nvar;

		for(j=0; j<countActiveU; j++)
		{
			i = energy -> activeU[j];
			label_bar = label_map[i];
			if(label_bar == label) is_active[offset+i] = true;
			else is_active[offset+i] = false;
		}

		// pairwise potentials
		int from, to;
		captype w1, w2, w3;
		//  0 w1
		// w2 w3
		for(j=0; j<countActiveP; j++)
		{
			i = energy->activeP[j];
			from = energy->pairIndex[i][0];	
			to = energy->pairIndex[i][1];
			w1 = energy->pairCost[i] * MIN(energy->truncation, abs(label-label_map[to]));
			w2 = energy->pairCost[i] * MIN(energy->truncation, abs(label_map[from]-label));
			w3 = energy->pairCost[i] * MIN(energy->truncation, abs(label_map[from]-label_map[to]));

			if(is_active[offset+from] && is_active[offset+to])
				g[label] -> edit_edge(nodes[offset+from], nodes[offset+to], 0, 0);
			else if(is_active[offset+from] && !is_active[offset+to])
			{
				tmp_tweight[to] -= w1;
				g[label] -> edit_edge(nodes[offset+from], nodes[offset+to], 0, 0);
			}
			else if(!is_active[offset+from] && is_active[offset+to])
			{
				tmp_tweight[from] -= w2;
				g[label] -> edit_edge(nodes[offset+from], nodes[offset+to], 0, 0);
			}
			else
			{
				if(label_map[from] == label_map[to])
					g[label] -> edit_edge(nodes[offset+from], nodes[offset+to], w2, w1);
				else
				{
					tmp_tweight[from] -= w3;
					w1 -= 0; w2 -= w3;

					if (w1 < 0)
					{
						tmp_tweight[from] += w1;
						tmp_tweight[to] -= w1;
						g[label] -> edit_edge(nodes[offset+from], nodes[offset+to], w1+w2, 0);
					}
					else if (w2 < 0)
					{
						tmp_tweight[from] -= w2;
						tmp_tweight[to] += w2;
						g[label] -> edit_edge(nodes[offset+from], nodes[offset+to], 0, w1+w2);
					}
					else
					{
						g[label] -> edit_edge(nodes[offset+from], nodes[offset+to], w2, w1);
					}
				}
			}
		}

		// unary potentials
		for(j=0; j<countActiveU; j++)
		{
			i = energy -> activeU[j];
			label_bar = label_map[i];
			if(label_bar == label)
			{
				constE += energy -> unaryCost[i][label];
				g[label] -> edit_tweights(nodes[offset+i], 0, 0);
			}
			else
			{
				// Editing the t-edge weights only once.
				// The changes due to pairwise terms are stored in tmp_weight[].
				if(tmp_tweight[i] >= 0)
					g[label] -> edit_tweights(nodes[offset+i], energy->unaryCost[i][label]+tmp_tweight[i], energy->unaryCost[i][label_bar]);
				else
					g[label] -> edit_tweights(nodes[offset+i], energy->unaryCost[i][label], energy->unaryCost[i][label_bar]-tmp_tweight[i]);
			}
		}

		E_old = E;
		clock_t start = clock();
		flowtype flowval = g[label] -> maxflow(true);
		comptime = (double(clock())-start);

		for(j=0; j<countActiveU; j++)
		{
			i = energy->activeU[j];
			tmplabel_map[i] = label_map[i];
			if (!is_active[offset+i] && g[label] -> what_segment(nodes[offset+i]) == Grapht::SINK)
				label_map[i] = label;
		}
		compute_energy();

		if (E >= E_old)
		{ // Accept the move only if the energy has decreased
			for(j=0; j<countActiveU; j++)
			{
				i = energy->activeU[j];
				label_map[i] = tmplabel_map[i];
			}
			E = E_old;
		}
		delete [] tmplabel_map;
		delete [] tmp_tweight;
		return comptime;
	}

	void compute_energy()
	{
		int i, j;

		E = 0;
		for (j=0; j<countActiveU; j++)
		{
			i = energy->activeU[j];
			if(label_map[i] == NOLABEL)
				E += MAX_EDGE_CAP;
			else
				E += energy->unaryCost[i][label_map[i]];
		}

		for(j=0; j<countActiveP; j++)
		{
			i = energy->activeP[j];
			if(label_map[energy->pairIndex[i][0]] != label_map[energy->pairIndex[i][1]])
				E += energy->pairCost[i];
		}
	}

	// To generate a random ordering of the labels
	void generate_permutation(unsigned char* lab_buf, int n)
	{
		int i, j;
        
		for (i=0; i<n; i++) lab_buf[i] = i;
		for (i=0; i<n-1; i++)
		{
			j = i + (int) (((double)rand()/RAND_MAX)*(n - i));
			int tmp = lab_buf[i]; lab_buf[i] = lab_buf[j]; lab_buf[j] = tmp;
		}
	}
};
