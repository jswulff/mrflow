#ifndef __ENERGY_H__
#define __ENERGY_H__

#include <cstdio>
#include <cmath>
#include "image.h"
#include "includes.h"

// Stores the energy corresponding to a graph

class Energy
{
	int nvar;			// No. of variables
	int npair;			// No. of pairwise terms
	captype truncation;
	int nlabel;			// No. of labels
	int countActiveU, countActiveP; // No. of 'active' unary&pairwise terms corr. to nodes unlabelled by Kovtun method
	int* activeU;			// List of active unary terms
	int* activeP;			// List of active pairwise terms
	captype **unaryCost;		// Unary cost

	// The cost of edge between the nodes (with IDs in) pairIndex[i][0] and pairIndex[i][1] is stored in pairCost[i]
	captype *pairCost;		// Pairwise cost
    	int **pairIndex;		// Pairwise terms index

	// Allocates memory for a grid topology graph grid_x X grid_y, and neighbourhood size nn(= 4 or 8)
	void gridTopology(int grid_x, int grid_y, int nn)
	{
		int count=0;
		for (int i=0;i<grid_x;i++)
			for(int j=0;j<grid_y;j++)
			{
				if (j<grid_y-1)
				{
					pairIndex[count][0] = i*grid_y + j;
					pairIndex[count][1] = i*grid_y + j + 1;
					count++;

					if (i>0 && nn==8)
					{
						pairIndex[count][0] = i*grid_y + j;
						pairIndex[count][1] = (i-1)*grid_y + j + 1;
						count++;
					}
				}

				if (i<grid_x-1)
				{
					pairIndex[count][0] = i*grid_y + j;
					pairIndex[count][1] = (i+1)*grid_y + j;
					count++;
					
					if (j<grid_y-1 && nn==8)
					{
						pairIndex[count][0] = i*grid_y + j;
						pairIndex[count][1] = (i+1)*grid_y + j + 1;
						count++;
					}
				}
			}
	}

	void allocateMemory()
	{
		unaryCost = new captype*[nvar];
		activeU = new int[nvar];
		for (int i=0;i<nvar;i++)
		{
			activeU[i]   = i;
			unaryCost[i] = new captype [nlabel];
            for (int j=0; j < nlabel; j++)
                unaryCost[i][j] = 0;
		}
		countActiveU = nvar;
		countActiveP = npair;

		
		pairIndex = new int*[npair];
		activeP = new int[npair];
		pairCost = new captype[npair];
		for (int i=0;i<npair;i++)
		{
			activeP[i]   = i;
			pairIndex[i] = new int[2];
            pairIndex[i][0] = 0;
            pairIndex[i][1] = 0;
		}
	}

public:
	int sameSmoothing;
	int unaryS, pairS;
	flowtype constantTerm;		// The constant energy term

	Energy(int nLabel, int trunc=1)
	{
		constantTerm = 0;
		sameSmoothing = 0;
		nlabel = nLabel;
		unaryS = 100;
		pairS = 0;
		truncation = trunc;
		nvar = 0;
		npair =0;
	}

	~Energy()
	{
		for (int i=0;i<nvar;i++)
			delete [] unaryCost[i];

		for (int i=0; i<npair; i++)
			delete [] pairIndex[i];

		delete [] activeU;
		delete [] activeP;
		delete [] unaryCost;
		delete [] pairCost;
		delete [] pairIndex;
	}

	// Read the energy values from a file
	// scale: Scales the unary and pairwise cost values
	Energy(const char *filename, int scale=1.0)
	{
		constantTerm = 0;
		truncation = 1;

		FILE *fp = fopen(filename,"r");
		int nLabel, nVar, nPair;

		fscanf(fp,"%d %d %d\n",&nVar,&nLabel,&nPair);

		nlabel = nLabel;
		nvar = nVar;
		npair = nPair;

		allocateMemory();
		int i, index, index1;
		double cost;

		// Extract Unary Terms
		for (i=0; i<nvar; i++)
		{
			fscanf(fp,"%d",&index);

			for(int j=0;j<nlabel;j++)
			{
				if (j==nlabel-1) fscanf(fp,"%lf \n",&cost);
				else fscanf(fp,"%lf ",&cost);
				unaryCost[i][j] = (captype) (cost*scale);
			}
		}

		// Extract pairwise terms
		for (i=0; i<npair; i++)
		{
			fscanf(fp,"%d %d %lf\n",&index, &index1, &cost);
			pairIndex[i][0] = index;
			pairIndex[i][1] = index1;
			pairCost[i] = (captype) (cost*scale);
		}
	}

	// Projects the energy function using the partial solution.
	// No new object is created (unlike Energy::Projection).
	void Project(unsigned char *partialSolution)
	{
		int i, j, k, indx0, indx1;
		int tmpCountP = 0, tmpCountU = 0;

		for(k=0; k<countActiveP; k++)
		{
			i = activeP[k];
			indx0 = pairIndex[i][0];
			indx1 = pairIndex[i][1];

			if (partialSolution[indx0]==NOLABEL && partialSolution[indx1]!=NOLABEL)
			{
				for (j=0;j<nlabel;j++)
					unaryCost[indx0][j] += pairCost[i]*MIN(truncation, abs(j-partialSolution[indx1]));
			}
			else if (partialSolution[indx0]!=NOLABEL && partialSolution[indx1]==NOLABEL) 
			{
				for (j=0;j<nlabel;j++)
					unaryCost[indx1][j] += pairCost[i]*MIN(truncation, abs(j-partialSolution[indx0]));
			}
			else if (partialSolution[indx0]==NOLABEL && partialSolution[indx1]==NOLABEL)
			{
				activeP[tmpCountP] = i;
				tmpCountP++;
			}
			else
				constantTerm += pairCost[i]*MIN(truncation, abs(partialSolution[indx0]-partialSolution[indx1]));
		}
		countActiveP = tmpCountP;

		for (k=0; k<countActiveU; k++)
		{
			i = activeU[k];
			if (partialSolution[i]==NOLABEL)
			{
				activeU[tmpCountU] = i;
				tmpCountU++;
			}
			else
				constantTerm += unaryCost[i][partialSolution[i]];
		}
		countActiveU = tmpCountU;
	}

	void printEnergy()
	{
		printf("\n Constant Term: %d\n",constantTerm);

		for (int i=0;i<nvar;i++)
		{
			printf("\nVar = %d ",i);
			for (int j=0;j<nlabel;j++)
				printf(" %d ",unaryCost[i][j]);
		}

		for (int i=0;i<npair;i++)
			printf("\n(%d,%d): %d",pairIndex[i][0],pairIndex[i][1],pairCost[i]);
	}

	void generate_unary()
	{
		for (int i=0;i<nvar;i++)
			for(int j=0;j<nlabel;j++)
				unaryCost[i][j] = captype(rand()%unaryS);
	}

	void generate_pairwise()
	{
		for (int i=0;i<npair;i++)
		{
			if (sameSmoothing) pairCost[i] = pairS;
			else
			{
			    if (pairS ==0) pairCost[i]=0;
			    else pairCost[i] = captype(rand()%pairS);
			}
		}
	}

	void set_truncation(int trunc)
	{
		assert(trunc<=nlabel);
		truncation = trunc;
	}

	captype computeEnergy(unsigned char *solution)
	{
		captype cost = constantTerm;

		for (int i=0;i<nvar;i++)
			cost += unaryCost[i][solution[i]];

		for (int i=0;i<npair;i++)
			cost += pairCost[i]*MIN(truncation,abs(solution[pairIndex[i][0]]-solution[pairIndex[i][1]]));

		return cost;
	}

	captype computeEnergy(int *solution)
	{
		captype cost = constantTerm;

		for (int i=0;i<nvar;i++)
			cost += unaryCost[i][solution[i]];

		for (int i=0;i<npair;i++)
			cost += pairCost[i]*MIN(truncation,abs(solution[pairIndex[i][0]]-solution[pairIndex[i][1]]));

		return cost;
	}

	void generateGridTopology(int grid_x, int grid_y, int nn)
	{
		assert(nn==4 || nn==8);
		nvar = grid_x * grid_y;
		if (nn<=4) npair = 2*grid_x*grid_y - (grid_x+grid_y);
		else npair = 4*grid_x*grid_y - 3*(grid_x+grid_y) + 2;
		allocateMemory();
		gridTopology(grid_x, grid_y, nn);
	}

	void printTopology()
	{
		for (int i=0;i<npair;i++)
			printf("(%d,%d)\n",pairIndex[i][0],pairIndex[i][1]);
	}

	// Creates a new Energy object, which is a projection of the current Energy object using a given partial solution
	Energy* Projection(unsigned char *partialSolution)
	{
		int i, j, k, indx0, indx1;
		flowtype conTerm=0;
		int remUnary=0, remPair=0;
		int *pairFlag = new int[npair];
		captype **projUnaryCost = new captype*[nvar];
		
		for (k=0; k<countActiveU; k++)
		{
			i = activeU[k];
			projUnaryCost[i] = new captype[nlabel];
			for (int j=0;j<nlabel;j++)
		        	projUnaryCost[i][j] = unaryCost[i][j];
		
			if (partialSolution[i]!=NOLABEL) 
				conTerm += unaryCost[i][partialSolution[i]];
			else remUnary ++;
		}

		for (k=0; k<countActiveP; k++)
		{
			i = activeP[k];
			pairFlag[i] = 0;
			indx0 = pairIndex[i][0];
			indx1 = pairIndex[i][1];
			
			if (partialSolution[indx0]==NOLABEL)
			{
				if (partialSolution[indx1]==NOLABEL)
				{
					pairFlag[i] = 1;
					remPair ++;
				}
				else
				{
					for (j=0; j<nlabel; j++)
						projUnaryCost[indx0][j] += pairCost[i]*MIN(truncation, abs(j-partialSolution[indx1]));
				}
			}
			else
			{
				if (partialSolution[indx1]==NOLABEL) 
				{
					for (j=0; j<nlabel; j++)
						projUnaryCost[indx1][j] += pairCost[i]*MIN(truncation, abs(j-partialSolution[indx0]));
				}
				else 
					conTerm += pairCost[i]*MIN(truncation, abs(partialSolution[indx0]-partialSolution[indx1]));
			}
		}

		Energy *project = new Energy(nlabel);
		project->nvar = remUnary;
		project->npair = remPair;
		project->allocateMemory();		
		project->constantTerm = conTerm;
		project->truncation = truncation;

		int *mapping = new int[nvar];
		int varCount=0;

		for(k=0; k<countActiveU; k++)
		{
			i = activeU[k];
			mapping[i] = -1;
			if (partialSolution[i]==NOLABEL)
			{
				mapping[i] = varCount;
				for (int j=0;j<nlabel;j++)
					project->unaryCost[varCount][j] = projUnaryCost[i][j];
				varCount++;
			}
		}

		int pairCount=0;
		for(k=0; k<countActiveP; k++)
		{
			i = activeP[k];
			if (pairFlag[i]==1)
			{
				project->pairCost[pairCount] = pairCost[i];
				project->pairIndex[pairCount][0] = mapping[pairIndex[i][0]];
				project->pairIndex[pairCount][1] = mapping[pairIndex[i][1]];
				pairCount++;
			}
		}

		// clean-up
		for (k=0; k<countActiveU; k++)
			delete projUnaryCost[k];

		delete projUnaryCost;
		delete pairFlag;
		delete mapping;

		return project;
	}

	friend class Kovtun;
	friend class image;
	friend class Aexpand;
	friend class TRWBP;
};

#endif
