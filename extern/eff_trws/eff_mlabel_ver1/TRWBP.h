#include "typePotts.h"
#include "MRFEnergy.h"
#include "energy.h"

// Computes the TRW and BP solutions for a given Energy e

class TRWBP
{

public:
	MRFEnergy<TypePotts>* mrf;
	MRFEnergy<TypePotts>::NodeId* nodes;
	MRFEnergy<TypePotts>::Options options;
	
	int *solutionT;
	int *solutionB;
	int nodeNum;
	int K;
	TypePotts::REAL *D;
	TypePotts::REAL energy, lowerBound;
	Energy *e;

	TRWBP(Energy *energy)
	{
		e = energy;
		nodeNum = e->nvar; // number of nodes
		K = e->nlabel; // number of labels
		D = new TypePotts::REAL[K];

		mrf = new MRFEnergy<TypePotts>(TypePotts::GlobalSize(K));
		nodes = new MRFEnergy<TypePotts>::NodeId[nodeNum];
		solutionT = new int[e->nvar];
		solutionB = new int[e->nvar];

		constructEnergy();
	}
	
	~TRWBP()
	{
		delete nodes;
		delete mrf;
		delete solutionT;
		delete solutionB;
	}

	void constructEnergy()
	{
		for (int i=0;i<e->nvar;i++)
		{
			for (int j=0;j<e->nlabel;j++)
				D[j] = e->unaryCost[i][j];
		
			nodes[i] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(D));
		}
	
		for (int i=0;i<e->npair;i++)
			mrf->AddEdge(nodes[e->pairIndex[i][0]], nodes[e->pairIndex[i][1]], TypePotts::EdgeData(e->pairCost[i]));
	}

	void printSolutions()
	{
		printf("\n TRW Solution: ");
		for (int i=0;i<e->nvar;i++)
			printf(" %d ", solutionT[i]);

		printf("\n BP Solution: ");
		for (int i=0;i<e->nvar;i++)
			printf(" %d ", solutionB[i]);
	}

	void minimize(bool debug = false, int maxIterT = 30, int maxIterB = 30)
	{
		// Function below is optional - it may help if, for example, nodes are added in a random order
		// mrf->SetAutomaticOrdering();

		/////////////////////// TRW-S algorithm //////////////////////
		options.m_iterMax = maxIterT; // maximum number of iterations
		if (!debug) options.m_printMinIter = options.m_iterMax;
		printf("\n\n");
		
		mrf->Minimize_TRW_S(options, lowerBound, energy);

		// read solution and compute its energy
		for (int i=0;i<e->nvar;i++)
			solutionT[i] = mrf->GetSolution(nodes[i]);

		//////////////////////// BP algorithm ////////////////////////
		mrf->ZeroMessages(); 
		options.m_iterMax = maxIterB;
		mrf->Minimize_BP(options, energy);

		for (int i=0;i<e->nvar;i++)
			solutionB[i] = mrf->GetSolution(nodes[i]);
	
		if (debug) 
			printf("\nTRW Solution energy: %d\tBP Solution energy: %d: const term: %d \n", e->computeEnergy(solutionT), e->computeEnergy(solutionB), e->constantTerm);
	}
};
