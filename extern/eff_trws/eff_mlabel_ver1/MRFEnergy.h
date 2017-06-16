/******************************************************************
Vladimir Kolmogorov, 2005
vnk@microsoft.com

(c) Microsoft Corporation. All rights reserved. 
*******************************************************************/

#ifndef __MRFENERGY_H__
#define __MRFENERGY_H__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include "typePotts.h"

// After MRFEnergy is allocated, there are two phases:
// 1. Energy construction. Only AddNode(), AddNodeData() and AddEdge() may be called.
// 
// Any call ZeroMessages(), SetAutomaticOrdering(), Minimize_TRW_S() or Minimize_BP()
// completes graph construction; MRFEnergy goes to the second phase:
// 2. Only functions AddNodeData(), ZeroMessages(), Minimize_TRW_S(), Minimize_BP()
// or GetSolution() may be called. (The last function can be called only after
// Minimize_TRW_S() or Minimize_BP()).


template <class T> class MRFEnergy
{
private:
	struct Node;

public:
	typedef typename T::Label      Label;
	typedef typename T::REAL       REAL;
	typedef typename T::GlobalSize GlobalSize;
	typedef typename T::LocalSize  LocalSize;
	typedef typename T::NodeData   NodeData;
	typedef typename T::EdgeData   EdgeData;

	typedef Node* NodeId;

	// Constructor.
	MRFEnergy(GlobalSize Kglobal);

	// Destructor.
	~MRFEnergy();

	// Error Function
	void ErrorFn(const char* msg);

	//////////////////////////////////////////////////////////
	//                 Energy construction                  //
	//////////////////////////////////////////////////////////

	// Adds a node with parameters K and data 
	// (see the corresponding message*.h file for description).
	// Note: information in data is copied into internal memory.
	// Cannot be called after energy construction is completed.
	NodeId AddNode(LocalSize K, NodeData data);

	// Modifies node parameter for existing node (namely, add information
	// in data to existing parameter). May be called at any time.
	// Node i must be NodeId returned by AddNode().
	void AddNodeData(NodeId i, NodeData data);

	// Adds an edge between i and j. data determins edge parameters
	// (see the corresponding message*.h file for description).
	// Note: information in data is copied into internal memory.
	// Cannot be called after energy construction is completed.
	void AddEdge(NodeId i, NodeId j, EdgeData data);

	//////////////////////////////////////////////////////////
	//                Energy construction end               //
	//////////////////////////////////////////////////////////

	// Clears all messages. Completes energy construction (if not completed yet).
	void ZeroMessages();

	// The algorithm depends on the order of nodes.
	// By default nodes are processed in the order in which they were added.
	// The function below permutes this order using certain heuristics.
	// It may speed up the algorithm if, for example, the original order is random.
	// 
	// Completes energy construction.
	// Cannot be called after energy construction is completed.
	void SetAutomaticOrdering();

	// The structure below specifies (1) stopping criteria and 
	// (2) how often to compute solution and print its energy.
	struct Options
	{
		Options()
		{
			// default parameters
			m_eps = -1; // not used
			m_iterMax = 1000000;
			m_printIter = 5;     // After 10 iterations start printing the lower bound
			m_printMinIter = 5; // and the energy every 5 iterations.
		}

		// stopping criterion
		REAL		m_eps; // stop if the increase in the lower bound during one iteration is less or equal than m_eps.
						   // Used only if m_eps >= 0, and only for TRW-S algorithm.
		int			m_iterMax; // maximum number of iterations

		// Option for printing lower bound and the energy.
		// Note: computing solution and its energy is slow
		// (it is comparable to the cost of one iteration).
		int		m_printIter; // print lower bound and energy every m_printIter iterations
		int		m_printMinIter; // do not print lower bound and energy before m_printMinIter iterations
	};

	// Returns number of iterations. Sets lowerBound and energy.
	int Minimize_TRW_S(Options& options, REAL& lowerBound, REAL& energy);

	// Returns number of iterations. Sets energy.
	int Minimize_BP(Options& options, REAL& energy);

	// Returns an integer in [0,Ki). Can be called only after Minimize().
	Label GetSolution(NodeId i);

	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
	//                   Implementation                     //
	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////
private:

	typedef typename T::Vector Vector;
	typedef typename T::Edge   Edge;

	struct MRFEdge;
	struct MallocBlock;

	MallocBlock*	m_mallocBlockFirst;
	Node*			m_nodeFirst;
	Node*			m_nodeLast;
	int				m_nodeNum;
	int				m_edgeNum;
	GlobalSize		m_Kglobal;
	int				m_vectorMaxSizeInBytes;

	bool			m_isEnergyConstructionCompleted;

	char*			m_buf; // buffer of size m_vectorMaxSizeInBytes 
				       //              + max(m_vectorMaxSizeInBytes, Edge::GetBufSizeInBytes(m_vectorMaxSizeInBytes))

	void CompleteGraphConstruction(); // nodes and edges cannot be added after calling this function
	void SetMonotonicTrees();

	REAL ComputeSolutionAndEnergy(); // sets Node::m_solution, returns value of the energy

	struct Node
	{
		int			m_ordering; // unique integer in [0,m_nodeNum-1)

		MRFEdge*	m_firstForward; // first edge going to nodes with greater m_ordering
		MRFEdge*	m_firstBackward; // first edge going to nodes with smaller m_ordering

		Node*		m_prev; // previous and next
		Node*		m_next; // nodes according to m_ordering

		Label		m_solution; // integer in [0,m_D.m_K)
		LocalSize	m_K; // local information about number of labels

		Vector		m_D; // must be the last member in the struct since its size is not fixed
	};

	struct MRFEdge
	{
		MRFEdge*	m_nextForward; // next forward edge with the same tail
		MRFEdge*	m_nextBackward; // next backward edge with the same head
		Node*		m_tail;
		Node*		m_head;

		REAL		m_gammaForward; // = rho_{ij} / rho_{i} where i=m_tail, j=m_head
		REAL		m_gammaBackward; // = rho_{ij} / rho_{j} where i=m_tail, j=m_head

		Edge		m_message; // must be the last member in the struct since its size is not fixed.
					           // Stores edge information and either forward or backward message.
					           // Most of the time it's the backward message; it gets replaced
					           // by the forward message only temporarily inside Minimize_TRW_S() and Minimize_BP().
	};

	// Use our own Malloc since 
	// (a) new in C++ is slow and allocates minimum memory of 64 bytes (in Visual C++)
	// (b) we want simple (one function) deallocation instead of going through every allocated element
	struct MallocBlock
	{
		static const int minBlockSizeInBytes = 4096 - 3*sizeof(void*);
		MallocBlock*	m_next;
		char*			m_current; // first element of available memory in this block
		char*			m_last; // first element outside of allocated memory for this block
	};
	char* Malloc(int bytesNum); 
};

template <class T> inline char* MRFEnergy<T>::Malloc(int bytesNum)
{
	if (!m_mallocBlockFirst || m_mallocBlockFirst->m_current+bytesNum > m_mallocBlockFirst->m_last)
	{
		int size = (bytesNum > MallocBlock::minBlockSizeInBytes) ? bytesNum : MallocBlock::minBlockSizeInBytes;
		MallocBlock* b = (MallocBlock*) new char[sizeof(MallocBlock) + size];
		if (!b) ErrorFn("Not enough memory");
		b->m_current = (char*) b + sizeof(MallocBlock);
		b->m_last    = b->m_current + size;

		b->m_next = m_mallocBlockFirst;
		m_mallocBlockFirst = b;
	}

	char* ptr = m_mallocBlockFirst->m_current;
	m_mallocBlockFirst->m_current += bytesNum;
	return ptr;
}

template <class T> inline typename T::Label MRFEnergy<T>::GetSolution(NodeId i)
{
	return i->m_solution;
}

template <class T> void MRFEnergy<T>::ErrorFn(const char* msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

template <class T> MRFEnergy<T>::MRFEnergy(GlobalSize Kglobal)
	: m_mallocBlockFirst(NULL),
	  m_nodeFirst(NULL),
	  m_nodeLast(NULL),
	  m_nodeNum(0),
	  m_edgeNum(0),
	  m_Kglobal(Kglobal),
	  m_vectorMaxSizeInBytes(0),
	  m_isEnergyConstructionCompleted(false),
	  m_buf(NULL)
{
}

template <class T> MRFEnergy<T>::~MRFEnergy<T>()
{
	while (m_mallocBlockFirst)
	{
		MallocBlock* next = m_mallocBlockFirst->m_next;
		delete m_mallocBlockFirst;
		m_mallocBlockFirst = next;
	}
}

template <class T> typename MRFEnergy<T>::NodeId MRFEnergy<T>::AddNode(LocalSize K, NodeData data)
{
	if (m_isEnergyConstructionCompleted)
	{
		ErrorFn("Error in AddNode(): graph construction completed - nodes cannot be added");
	}

	int actualVectorSize = Vector::GetSizeInBytes(m_Kglobal, K);
	if (actualVectorSize < 0)
	{
		ErrorFn("Error in AddNode() (invalid parameter?)");
	}
	if (m_vectorMaxSizeInBytes < actualVectorSize)
	{
		m_vectorMaxSizeInBytes = actualVectorSize;
	}
	int nodeSize = sizeof(Node) - sizeof(Vector) + actualVectorSize;
	Node* i = (Node *) Malloc(nodeSize);

	i->m_K = K;
	i->m_D.Initialize(m_Kglobal, K, data);

	i->m_firstForward = NULL;
	i->m_firstBackward = NULL;
	i->m_prev = m_nodeLast;
	if (m_nodeLast)
	{
		m_nodeLast->m_next = i;
	}
	else
	{
		m_nodeFirst = i;
	}
	m_nodeLast = i;
	i->m_next = NULL;

	i->m_ordering = m_nodeNum ++;

	return i;
}

template <class T> void MRFEnergy<T>::AddNodeData(NodeId i, NodeData data)
{
	i->m_D.Add(m_Kglobal, i->m_K, data);
}

template <class T> void MRFEnergy<T>::AddEdge(NodeId i, NodeId j, EdgeData data)
{
	if (m_isEnergyConstructionCompleted)
	{
		ErrorFn("Error in AddNode(): graph construction completed - nodes cannot be added");
	}

	MRFEdge* e;

	int actualEdgeSize = Edge::GetSizeInBytes(m_Kglobal, i->m_K, j->m_K, data);
	if (actualEdgeSize < 0)
	{
		ErrorFn("Error in AddEdge() (invalid parameter?)");
	}
	int MRFedgeSize = sizeof(MRFEdge) - sizeof(Edge) + actualEdgeSize;
	e = (MRFEdge*) Malloc(MRFedgeSize);

	e->m_message.Initialize(m_Kglobal, i->m_K, j->m_K, data, &i->m_D, &j->m_D);

	e->m_tail = i;
	e->m_nextForward = i->m_firstForward;
	i->m_firstForward = e;

	e->m_head = j;
	e->m_nextBackward = j->m_firstBackward;
	j->m_firstBackward = e;

	m_edgeNum ++;
}

/////////////////////////////////////////////////////////////////////////////////

template <class T> void MRFEnergy<T>::ZeroMessages()
{
	Node* i;
	MRFEdge* e;

	if (!m_isEnergyConstructionCompleted)
	{
		CompleteGraphConstruction();
		return;
	}

	for (i=m_nodeFirst; i; i=i->m_next)
	{
		for (e=i->m_firstForward; e; e=e->m_nextForward)
		{
			e->m_message.GetMessagePtr()->SetZero(m_Kglobal, i->m_K);
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////

template <class T> void MRFEnergy<T>::CompleteGraphConstruction()
{
	Node* i;
	Node* j;
	MRFEdge* e;
	MRFEdge* ePrev;

	if (m_isEnergyConstructionCompleted)
	{
		ErrorFn("Fatal error in CompleteGraphConstruction");
	}

	printf("Completing graph construction... ");

	if (m_buf)
	{
		ErrorFn("CompleteGraphConstruction(): fatal error");
	}
	m_buf = (char *) Malloc(m_vectorMaxSizeInBytes + 
		m_vectorMaxSizeInBytes > Edge::GetBufSizeInBytes(m_vectorMaxSizeInBytes) ?
		m_vectorMaxSizeInBytes : Edge::GetBufSizeInBytes(m_vectorMaxSizeInBytes) );

	// set forward and backward edges properly
#ifdef _DEBUG
	int ordering;
	for (i=m_nodeFirst, ordering=0; i; i=i->m_next, ordering++)
	{
		if ( (i->m_ordering != ordering)
		  || (i->m_ordering == 0 && i->m_prev)
		  || (i->m_ordering != 0 && i->m_prev->m_ordering != ordering-1) )
		{
			ErrorFn("CompleteGraphConstruction(): fatal error (wrong ordering)");
		}
	}
	if (ordering != m_nodeNum)
	{
		ErrorFn("CompleteGraphConstruction(): fatal error");
	}
#endif
	for (i=m_nodeFirst; i; i=i->m_next)
	{
		i->m_firstBackward = NULL;
	}
	for (i=m_nodeFirst; i; i=i->m_next)
	{
		ePrev = NULL;
		for (e=i->m_firstForward; e; )
		{
			assert(i == e->m_tail);
			j = e->m_head;

			if (i->m_ordering < j->m_ordering)
			{
				e->m_nextBackward = j->m_firstBackward;
				j->m_firstBackward = e;

				ePrev = e;
				e = e->m_nextForward;
			}
			else
			{
				e->m_message.Swap(m_Kglobal, i->m_K, j->m_K);
				e->m_tail = j;
				e->m_head = i;

				MRFEdge* eNext = e->m_nextForward;

				if (ePrev)
				{
					ePrev->m_nextForward = e->m_nextForward;
				}
				else
				{
					i->m_firstForward = e->m_nextForward;
				}

				e->m_nextForward = j->m_firstForward;
				j->m_firstForward = e;

				e->m_nextBackward = i->m_firstBackward;
				i->m_firstBackward = e;

				e = eNext;
			}
		}
	}

	m_isEnergyConstructionCompleted = true;

	ZeroMessages();

	printf("done\n");
}

/////////////////////////////////////////////////////////////////////////////////

template <class T> void MRFEnergy<T>::SetAutomaticOrdering()
{
	int dMin;
	Node* i;
	Node* iMin;
	Node* list;
	Node* listBoundary;
	MRFEdge* e;

	if (m_isEnergyConstructionCompleted)
	{
		ErrorFn("Error in SetAutomaticOrdering(): function cannot be called after graph construction is completed");
	}

	printf("Setting automatic ordering... ");

	list = m_nodeFirst;
	listBoundary = NULL;
	m_nodeFirst = m_nodeLast = NULL;
	for (i=list; i; i=i->m_next)
	{
		i->m_ordering = 2*m_nodeNum; // will contain remaining degree mod m_nodeNum (i.e. number of edges connecting to nodes in 'listBoundary' and 'list')
		                             // if i->m_ordering \in [2*m_nodeNum;  3*m_nodeNum) - not assigned yet, belongs to 'list'
		                             // if i->m_ordering \in [m_nodeNum;    2*m_nodeNum) - not assigned yet, belongs to 'listBoundary'
		                             // if i->m_ordering \in [0;            m_nodeNum  ) - assigned, belongs to 'm_nodeFirst'
		for (e=i->m_firstForward; e; e=e->m_nextForward)
		{
			i->m_ordering ++;
		}
		for (e=i->m_firstBackward; e; e=e->m_nextBackward)
		{
			i->m_ordering ++;
		}
	}

	while (list)
	{
		// find node with the smallest remaining degree in list
		dMin = m_nodeNum;
		for (i=list; i; i=i->m_next)
		{
			assert(i->m_ordering >= 2*m_nodeNum);
			if (dMin > i->m_ordering - 2*m_nodeNum)
			{
				dMin = i->m_ordering - 2*m_nodeNum;
				iMin = i;
			}
		}
		i = iMin;

		// remove i from list
		if (i->m_prev) i->m_prev->m_next = i->m_next;
		else           list              = i->m_next;
		if (i->m_next) i->m_next->m_prev = i->m_prev;

		// add i to listBoundary
		listBoundary = i;
		i->m_prev = NULL;
		i->m_next = NULL;
		i->m_ordering -= m_nodeNum;

		while (listBoundary)
		{
			// find node with the smallest remaining degree in listBoundary
			dMin = m_nodeNum;
			for (i=listBoundary; i; i=i->m_next)
			{
				assert(i->m_ordering >= m_nodeNum && i->m_ordering < 2*m_nodeNum);
				if (dMin > i->m_ordering - m_nodeNum)
				{
					dMin = i->m_ordering - m_nodeNum;
					iMin = i;
				}
			}
			i = iMin;

			// remove i from listBoundary
			if (i->m_prev) i->m_prev->m_next = i->m_next;
			else           listBoundary      = i->m_next;
			if (i->m_next) i->m_next->m_prev = i->m_prev;

			// add i to m_nodeFirst
			if (m_nodeLast)
			{
				m_nodeLast->m_next = i;
				i->m_ordering = m_nodeLast->m_ordering + 1;
			}
			else
			{
				m_nodeFirst = i;
				i->m_ordering = 0;
			}
			i->m_prev = m_nodeLast;
			m_nodeLast = i;
			i->m_next = NULL;

			// process neighbors of i=m_nodeLast: decrease their remaining degree,
			// put them into listBoundary (if they are in list)
			for (e=m_nodeLast->m_firstForward; e; e=e->m_nextForward)
			{
				assert(m_nodeLast == e->m_tail);
				i = e->m_head;
				if (i->m_ordering >= m_nodeNum)
				{
					i->m_ordering --; // decrease remaining degree of i
					if (i->m_ordering >= 2*m_nodeNum)
					{
						// remove i from list
						if (i->m_prev) i->m_prev->m_next = i->m_next;
						else           list              = i->m_next;
						if (i->m_next) i->m_next->m_prev = i->m_prev;

						// add i to listBoundary
						if (listBoundary) listBoundary->m_prev = i;
						i->m_prev = NULL;
						i->m_next = listBoundary;
						listBoundary = i;
						i->m_ordering -= m_nodeNum;
					}
				}
			}
			for (e=m_nodeLast->m_firstBackward; e; e=e->m_nextBackward)
			{
				assert(m_nodeLast == e->m_head);
				i = e->m_tail;
				if (i->m_ordering >= m_nodeNum)
				{
					i->m_ordering --; // decrease remaining degree of i
					if (i->m_ordering >= 2*m_nodeNum)
					{
						// remove i from list
						if (i->m_prev) i->m_prev->m_next = i->m_next;
						else           list              = i->m_next;
						if (i->m_next) i->m_next->m_prev = i->m_prev;

						// add i to listBoundary
						if (listBoundary) listBoundary->m_prev = i;
						i->m_prev = NULL;
						i->m_next = listBoundary;
						listBoundary = i;
						i->m_ordering -= m_nodeNum;
					}
				}
			}
		}
	}

	printf("done\n");

	CompleteGraphConstruction();
}

/////////////////////////////////////////////////////////////////////////////////

template <class T> void MRFEnergy<T>::SetMonotonicTrees()
{
	Node* i;
	MRFEdge* e;

	if (!m_isEnergyConstructionCompleted)
	{
		CompleteGraphConstruction();
	}

	for (i=m_nodeFirst; i; i=i->m_next)
	{
		REAL mu;

		int nForward = 0, nBackward = 0;
		for (e=i->m_firstForward; e; e=e->m_nextForward)
		{
			nForward ++;
		}
		for (e=i->m_firstBackward; e; e=e->m_nextBackward)
		{
			nBackward ++;
		}
		int ni = (nForward > nBackward) ? nForward : nBackward;

		mu = (REAL)1 / ni;
		for (e=i->m_firstBackward; e; e=e->m_nextBackward)
		{
			e->m_gammaBackward = mu;
		}
		for (e=i->m_firstForward; e; e=e->m_nextForward)
		{
			e->m_gammaForward = mu;
		}
	}
}

template <class T> int MRFEnergy<T>::Minimize_TRW_S(Options& options, REAL& lowerBound, REAL& energy)
{ 

	double start = clock(); 

	Node* i;
	Node* j;
	MRFEdge* e;
	REAL vMin;
	int iter;
	REAL lowerBoundPrev;

	if (!m_isEnergyConstructionCompleted)
	{
		CompleteGraphConstruction();
	}

	printf("TRW_S algorithm\n");

	SetMonotonicTrees();

	Vector* Di = (Vector*) m_buf;
	void* buf = (void*) (m_buf + m_vectorMaxSizeInBytes);

	iter = 0;

	// main loop
	for (iter=1; ; iter++)
	{
		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		for (i=m_nodeFirst; i; i=i->m_next)
		{
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e=i->m_firstForward; e; e=e->m_nextForward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}
			for (e=i->m_firstBackward; e; e=e->m_nextBackward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}

			// normalize Di, update lower bound
			// vMin = Di->ComputeAndSubtractMin(m_Kglobal, i->m_K); // do not compute lower bound
			// lowerBound += vMin;                                  // during the forward pass

			// pass messages from i to nodes with higher m_ordering
			for (e=i->m_firstForward; e; e=e->m_nextForward)
			{
				assert(e->m_tail == i);
				j = e->m_head;

				vMin = e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, e->m_gammaForward, 0, buf);

				// lowerBound += vMin; // do not compute lower bound during the forward pass
			}
		}

		////////////////////////////////////////////////
		//               backward pass                //
		////////////////////////////////////////////////
		lowerBound = 0;

		for (i=m_nodeLast; i; i=i->m_prev)
		{
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e=i->m_firstBackward; e; e=e->m_nextBackward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}
			for (e=i->m_firstForward; e; e=e->m_nextForward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}

			// normalize Di, update lower bound
			vMin = Di->ComputeAndSubtractMin(m_Kglobal, i->m_K);
			lowerBound += vMin;

			// pass messages from i to nodes with smaller m_ordering
			for (e=i->m_firstBackward; e; e=e->m_nextBackward)
			{
				assert(e->m_head == i);
				j = e->m_tail;

				vMin = e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, e->m_gammaBackward, 1, buf);

				lowerBound += vMin;
			}
		}

		////////////////////////////////////////////////
		//          check stopping criterion          //
		////////////////////////////////////////////////
		bool finishFlag = false;
		if (iter >= options.m_iterMax)
		{
			finishFlag = true;
		}
		if (options.m_eps >= 0)
		{
			if (iter > 1 && lowerBound - lowerBoundPrev <= options.m_eps)
			{
				finishFlag = true;
			}
			lowerBoundPrev = lowerBound;
		}

		// print lower bound and energy, if necessary
		if (  
		finishFlag || 
			( iter>=options.m_printMinIter && 
			(options.m_printIter<1 || iter%options.m_printIter==0) )
		)
		{
			energy = ComputeSolutionAndEnergy();
			printf("iter %d: lower bound = %f, energy = %f  time = %f \n", iter, lowerBound, energy,(clock()-start)/CLOCKS_PER_SEC);
			//printf("%f  %f  %f \n", (clock()-start)/CLOCKS_PER_SEC, lowerBound, energy);
		}

		// if finishFlag==true terminate
		if (finishFlag)
		{
			break;
		}
	}

	return iter;
}

template <class T> int MRFEnergy<T>::Minimize_BP(Options& options, REAL& energy)
{
	Node* i;
	Node* j;
	MRFEdge* e;
	REAL vMin;
	int iter;

	double start = clock(); 

	if (!m_isEnergyConstructionCompleted)
	{
		CompleteGraphConstruction();
	}

	printf("BP algorithm\n");

	Vector* Di = (Vector*) m_buf;
	void* buf = (void*) (m_buf + m_vectorMaxSizeInBytes);

	iter = 0;

	// main loop
	for (iter=1; ; iter++)
	{
		////////////////////////////////////////////////
		//                forward pass                //
		////////////////////////////////////////////////
		for (i=m_nodeFirst; i; i=i->m_next)
		{
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e=i->m_firstForward; e; e=e->m_nextForward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}
			for (e=i->m_firstBackward; e; e=e->m_nextBackward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}

			// pass messages from i to nodes with higher m_ordering
			for (e=i->m_firstForward; e; e=e->m_nextForward)
			{
				assert(i == e->m_tail);
				j = e->m_head;

				const REAL gamma = 1;

				e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, gamma, 0, buf);
			}
		}

		////////////////////////////////////////////////
		//               backward pass                //
		////////////////////////////////////////////////

		for (i=m_nodeLast; i; i=i->m_prev)
		{
			Di->Copy(m_Kglobal, i->m_K, &i->m_D);
			for (e=i->m_firstBackward; e; e=e->m_nextBackward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}
			for (e=i->m_firstForward; e; e=e->m_nextForward)
			{
				Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
			}

			// pass messages from i to nodes with smaller m_ordering
			for (e=i->m_firstBackward; e; e=e->m_nextBackward)
			{
				assert(i == e->m_head);
				j = e->m_tail;

				const REAL gamma = 1;

				vMin = e->m_message.UpdateMessage(m_Kglobal, i->m_K, j->m_K, Di, gamma, 1, buf);
			}
		}

		////////////////////////////////////////////////
		//          check stopping criterion          //
		////////////////////////////////////////////////
		bool finishFlag = false;
		if (iter >= options.m_iterMax)
		{
			finishFlag = true;
		}

		// print energy, if necessary
		if (  
		finishFlag || 
			( iter>=options.m_printMinIter && 
			(options.m_printIter<1 || iter%options.m_printIter==0) )
		)
		{
			energy = ComputeSolutionAndEnergy();
			//printf("iter %d: energy = %f time = %f\n", iter, energy, (clock()-start)/CLOCKS_PER_SEC);
			printf("%f %f\n", (clock()-start)/CLOCKS_PER_SEC, energy);
		}

		// if finishFlag==true terminate
		if (finishFlag)
		{
			break;
		}
	}

	return iter;
}

template <class T> typename T::REAL MRFEnergy<T>::ComputeSolutionAndEnergy()
{
	Node* i;
	Node* j;
	MRFEdge* e;
	REAL E = 0;

	Vector* DiBackward = (Vector*) m_buf; // cost of backward edges plus Di at the node
	Vector* Di = (Vector*) (m_buf + m_vectorMaxSizeInBytes); // all edges plus Di at the node

	for (i=m_nodeFirst; i; i=i->m_next)
	{
		// Set Ebackward[ki] to be the sum of V(ki,j->m_solution) for backward edges (i,j).
		// Set Di[ki] to be the value of the energy corresponding to
		// part of the graph considered so far, assuming that nodes u
		// in this subgraph are fixed to u->m_solution

		DiBackward->Copy(m_Kglobal, i->m_K, &i->m_D);
		for (e=i->m_firstBackward; e; e=e->m_nextBackward)
		{
			assert(i == e->m_head);
			j = e->m_tail;

			e->m_message.AddColumn(m_Kglobal, j->m_K, i->m_K, j->m_solution, DiBackward, 0);
		}

		// add forward edges
		Di->Copy(m_Kglobal, i->m_K, DiBackward);

		for (e=i->m_firstForward; e; e=e->m_nextForward)
		{
			Di->Add(m_Kglobal, i->m_K, e->m_message.GetMessagePtr());
		}

		Di->ComputeMin(m_Kglobal, i->m_K, i->m_solution);

		// update energy
		E += DiBackward->GetValue(m_Kglobal, i->m_K, i->m_solution);
	}

	return E;
}

#endif
