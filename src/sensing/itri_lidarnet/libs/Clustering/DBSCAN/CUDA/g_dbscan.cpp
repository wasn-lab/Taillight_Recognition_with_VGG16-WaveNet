#include "g_dbscan.h"

using namespace pcl;

bool has_nonzero(std::vector<int>& v)
{
  for (int i : v)
  {
    if (v[i] > 0)
    {
      return true;
    }
  }
  return false;
}

GDBSCAN::GDBSCAN(const Dataset::Ptr &dset)
  : m_dset(dset)
  , d_data(nullptr)
  , d_label(nullptr)
  , vA_size(sizeof(int) * dset->rows())
  , d_Va0(nullptr)
  , d_Va1(nullptr)
  , h_Va0(dset->rows(), 0)
  , h_Va1(dset->rows(), 0)
  , d_Ea(nullptr)
  , d_Fa(nullptr)
  , d_Xa(nullptr)
  , d_eps(nullptr)
  , core(dset->rows(), false)
  , labels(dset->rows(), -1)
  , cluster_id(0)
{
  // std::cout << "number of points: " << m_dset->num_points() << std::endl;
  // std::cout << "vA_size: " << vA_size << std::endl;
  ErrorHandle(cudaMallocManaged(reinterpret_cast<void**>(&d_data), sizeof(float) * m_dset->num_points()), "d_data");
  ErrorHandle(cudaMallocManaged(reinterpret_cast<void**>(&d_label), sizeof(int) * m_dset->rows()), "d_label");
  ErrorHandle(cudaMallocManaged(reinterpret_cast<void**>(&d_Va0), vA_size), "d_Va0");
  ErrorHandle(cudaMallocManaged(reinterpret_cast<void**>(&d_Va1), vA_size), "d_Va1");
  ErrorHandle(cudaMallocManaged(reinterpret_cast<void**>(&d_Fa), vA_size), "d_Fa");
  ErrorHandle(cudaMallocManaged(reinterpret_cast<void**>(&d_Xa), vA_size), "d_Xa");
  ErrorHandle(cudaMallocManaged(reinterpret_cast<void**>(&d_eps), sizeof(float) * 5), "d_eps");

  ErrorHandle(cudaMemcpy(d_label, &(m_dset->label()[0]), sizeof(int) * m_dset->rows(), cudaMemcpyHostToDevice), "memcpy"
                                                                                                                " of "
                                                                                                                "d_"
                                                                                                                "labe"
                                                                                                                "l");

  size_t copysize = m_dset->cols() * sizeof(float);

  for (size_t i = 0; i < m_dset->rows(); ++i)
  {
    ErrorHandle(cudaMemcpy(d_data + i * m_dset->cols(), m_dset->data()[i].data(), copysize, cudaMemcpyHostToDevice),
                "memcpy");

    // std::cout << "Copied " << i << "th row to device, size = " << copysize;
  }
}

GDBSCAN::~GDBSCAN()
{
  if (d_data != nullptr)
  {
    cudaFree(d_data);
    d_data = nullptr;
  }

  if (d_label != nullptr)
  {
    cudaFree(d_label);
    d_label = nullptr;
  }

  if (d_Va0 != nullptr)
  {
    cudaFree(d_Va0);
    d_Va0 = nullptr;
  }

  if (d_Va1 != nullptr)
  {
    cudaFree(d_Va1);
    d_Va1 = nullptr;
  }

  if (d_Ea != nullptr)
  {
    cudaFree(d_Ea);
    d_Ea = nullptr;
  }

  if (d_Fa != nullptr)
  {
    cudaFree(d_Fa);
    d_Fa = nullptr;
  }

  if (d_Xa != nullptr)
  {
    cudaFree(d_Xa);
    d_Xa = nullptr;
  }

  if (d_eps != nullptr)
  {
    cudaFree(d_eps);
    d_eps = nullptr;
  }
}

void GDBSCAN::fit(float* eps, const size_t* min_elems, int maxThreadsNumber)
{
  // Vertices degree calculation: For each vertex, we calculate the
  // total number of adjacent vertices. However we can use the multiple cores of
  // the GPU to process multiple vertices in parallel. Our parallel strategy
  // using GPU assigns a thread to each vertex, i.e., each entry of the vector
  // Va. Each GPU thread will count how many adjacent vertex has under its
  // responsibility, filling the first value on the vector Va. As we can see,
  // there are no dependency (or communication) between those parallel tasks
  // (embarrassingly parallel problem). Thus, the computational complexity can
  // be reduced from O(V2) to O(V).

  // Determine which mode we are in. If the input column size is 5(XYZIL), we are in label mode
  int label_mode = 0;
  auto n = static_cast<int>(m_dset->rows());        // size()
  auto colsize = static_cast<int>(m_dset->cols());  // 3 XYZ
  // 2020/4/29
  // for label-based G-DBSCAN, the cols() might be 5 XYZIL
  if (colsize == 5)
  {
    label_mode = 1;
    std::cout << "[DBSCAN] In label mode" << std::endl;
  }

  ErrorHandle(cudaMemcpy(d_eps, &eps[0], 5 * sizeof(float), cudaMemcpyHostToDevice), "memcpy of eps from host to "
                                                                                     "device");

  vertdegree(n, colsize, d_eps, d_data, d_Va0, d_label, maxThreadsNumber, label_mode);
  // std::cout << "[DBSCAN] vertdegree successed" << std::endl;
  // std::cout << "Executed vertdegree transfer";

  // Calculation of the adjacency lists indices: The second value in Va is related to the start
  // index in Ea of the adjacency list of a particular vertex. The calculation
  // of this value depends on the start index of the vertex adjacency list and
  // the degree of the previous vertex. For example, the start index for the
  // vertex 0 is 0, since it is the first vertex. For the vertex 1, the start
  // index is the start index from the previous vertex (i.e. 0), plus its
  // degree, already calculated in the previous step. We realize that we have a
  // data dependency where the next vertex depends on the calculation of the
  // preceding vertices. This is a problem that can be efficiently done in
  // parallel using an exclusive scan operation [23]. For this operation, we
  // used the thrust library, distributed as part of the CUDA SDK. This library
  // provides, among others algorithms, an optimized exclusive scan
  // implementation that is suitable for our method
  // ErrorHandle(cudaMemcpy(&h_Va0[0], d_Va0, vA_size, cudaMemcpyDeviceToHost), "memcpy Va0 device to host");
  // std::cout << "size of h_Va0:" << sizeof(h_Va0)/sizeof(int) << std::endl;
  adjlistsind(n, d_Va0, d_Va1);
  // std::cout << "[DBSCAN] adjlistsind successed" << std::endl;
  // Executed adjlistsind transfer;

  ErrorHandle(cudaMemcpy(&h_Va0[0], d_Va0, vA_size, cudaMemcpyDeviceToHost), "memcpy Va0 device to host");
  ErrorHandle(cudaMemcpy(&h_Va1[0], d_Va1, vA_size, cudaMemcpyDeviceToHost), "memcpy Va1 device to host");

  // Finished transfer;
  for (int i = 0; i < n; ++i)
  {
    if (m_dset->cols() > 3)
    {
      auto pointclass_id = (int)(m_dset->data()[i][4]);
      if (static_cast<size_t>(h_Va0[i]) >= min_elems[pointclass_id])
      {
        core[i] = true;
      }
    }
    else
    {
      if (static_cast<size_t>(h_Va0[i]) >= min_elems[0])
      {
        core[i] = true;
      }
    }
  }

  // Assembly of adjacency lists: Having the vector Va been completely filled, i.e., for each
  // vertex, we know its degree and the start index of its adjacency list,
  // calculated in the two previous steps, we can now simply mount the compact
  // adjacency list, represented by Ea. Following the logic of the first step,
  // we assign a GPU thread to each vertex. Each of these threads will fill the
  // adjacency list of its associated vertex with all vertices adjacent to it.
  // The adjacency list for each vertex starts at the indices present in the
  // second value of Va, and has an offset related to the degree of the vertex.

  size_t ea_size;
  if (!h_Va0.empty())
  {
    ea_size = static_cast<size_t>(h_Va0[h_Va0.size() - 1] + h_Va1[h_Va1.size() - 1]) * sizeof(int);
  }
  else
  {
    ea_size = 0;
  }
  // std::cout << "Allocating " << Ea_size << " bytes for Ea "<< h_Va0[h_Va0.size() - 1] << "+" << h_Va1[h_Va1.size() -
  // 1];

  if (d_Ea != nullptr)
  {
    cudaFree(d_Ea);
    d_Ea = nullptr;
  }

  ErrorHandle(cudaMalloc(reinterpret_cast<void**>(&d_Ea), ea_size), "d_Ea malloc");
  // std::cout << "[DBSCAN] before asmadjlist" << std::endl;
  asmadjlist(n, colsize, d_eps, d_data, d_Va1, d_Ea, d_label, label_mode);
  // std::cout << "[DBSCAN] asmadjlist successed" << std::endl;
}

void GDBSCAN::breadth_first_search(int i, int32_t cluster, std::vector<bool>& visited)
{
  auto n = static_cast<int>(m_dset->rows());

  std::vector<int> xa(m_dset->rows(), 0);
  std::vector<int> fa(m_dset->rows(), 0);
  // int Xa = new int[m_dset->rows()];
  // int Fa = new int[m_dset->rows()];

  fa[i] = 1;
  // std::cout << "size of Fa: " << m_dset->rows() * sizeof(int) << std::endl;
  // std::cout << "vA_size: " << vA_size << std::endl;
  // Fa_Xa_to_device;
  ErrorHandle(cudaMemcpy(d_Fa, &fa[0], vA_size, cudaMemcpyHostToDevice), "memcpy Fa host to device");
  ErrorHandle(cudaMemcpy(d_Xa, &xa[0], vA_size, cudaMemcpyHostToDevice), "memcpy Xa host to device");

  while (has_nonzero(fa))
  {
    breadth_first_search_kern(n, d_Ea, d_Va0, d_Va1, d_Fa, d_Xa);
    // Fa_to_host;
    ErrorHandle(cudaMemcpy(&fa[0], d_Fa, vA_size, cudaMemcpyDeviceToHost), "memcpy Fa device to host");
  }

  // Xa_to_host;
  ErrorHandle(cudaMemcpy(&xa[0], d_Xa, vA_size, cudaMemcpyDeviceToHost), "memcpy Xa device to host");

  for (size_t i = 0; i < m_dset->rows(); ++i)
  {
    if (xa[i] != 0)
    {
      visited[i] = true;
      labels[i] = cluster;
    }
  }
}

void GDBSCAN::ErrorHandle(cudaError_t r, const std::string &Msg)
{
  if (r != cudaSuccess)
  {
    throw std::runtime_error("[DBSCAN] CUDA Error :" + Msg + ", " + std::to_string(r));
  }
}

void GDBSCAN::predict(pcl::IndicesClusters& index)
{
  // For this step, we decided to parallelize the BFS. Our parallelization
  // approach in CUDA is based on the work presented in [22], which performs a
  // level synchronization, i.e. the BFS traverses the graph in levels. Once a
  // level is visited, it is not visited again. The concept of border in the BFS
  // corresponds to all nodes being processed at the current level. In our
  // implementation we assign one thread to each vertex. Two Boolean vectors,
  // Borders and Visiteds, namely Fa and Xa, respectively, of size V are created
  // to store the vertices that are on the border of BFS (vertices of the
  // current level) and the vertices already visited. In each iteration, each
  // thread (vertex) looks for its entry in the vector Fa. If its position is
  // marked, the vertex removes its own entry on Fa and marks its position in
  // the vector Xa (it is removed from the border, and it has been visited, so
  // we can go to the next level). It also adds its neighbours to the vector Fa
  // if they have not already been visited, thus beginning the search in a new
  // level. This process is repeated until the boundary becomes empty. We
  // illustrate the functioning of our BFS parallel implementation in Algorithm
  // 3 and 4.

  cluster_id = 0;
  std::vector<bool> visited(m_dset->rows(), false);

  for (size_t i = 0; i < m_dset->rows(); ++i)
  {
    if (visited[i])
    {
      continue;
    }
    if (!core[i])
    {
      continue;
    }

    visited[i] = true;
    labels[i] = cluster_id;
    breadth_first_search(static_cast<int>(i), cluster_id, visited);
    cluster_id += 1;
  }

  if (cluster_id > 0)
  {
    PointIndices buff[cluster_id];

    for (size_t i = 0; i < labels.size(); i++)  // scan all points
    {
      if (labels.at(i) >= 0)
      {
        buff[labels.at(i)].indices.push_back(i);
      }
    }

    for (int k = 0; k < cluster_id; k++)
    {
      index.push_back(buff[k]);
    }
  }
  else
  {
    index.resize(0);
  }
}
