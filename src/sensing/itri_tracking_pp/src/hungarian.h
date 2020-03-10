#ifndef __HUNGARIAN_H__
#define __HUNGARIAN_H__

#include "tpp.h"
#include <iostream>
#include <vector>

#include <cstdlib>
#include <cfloat>  // for DBL_MAX
#include <cmath>   // for fabs()

using namespace std;

namespace tpp
{
class Hungarian
{
public:
  Hungarian();
  ~Hungarian();
  double solve(vector<vector<double> >& DistMatrix, vector<int>& Assignment);

private:
  DISALLOW_COPY_AND_ASSIGN(Hungarian);

  void assignment_optimal(int* assignment, double* cost, double* distMatrix, int nOfRows, int nOfColumns);
  void build_assignment_vector(int* assignment, bool* starMatrix, int nOfRows, int nOfColumns);
  void compute_assignment_cost(int* assignment, double* cost, double* distMatrix, int nOfRows);
  void step2a(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
              bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
  void step2b(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
              bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
  void step3(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
  void step4(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
  void step5(int* assignment, double* distMatrix, bool* starMatrix, bool* newStarMatrix, bool* primeMatrix,
             bool* coveredColumns, bool* coveredRows, int nOfRows, int nOfColumns, int minDim);
};
}  // namespace tpp

#endif  // __HUNGARIAN_H__