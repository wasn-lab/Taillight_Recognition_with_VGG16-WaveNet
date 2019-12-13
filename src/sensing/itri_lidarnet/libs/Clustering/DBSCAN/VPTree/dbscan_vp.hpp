#ifndef DBSCAN_VP_HPP
#define DBSCAN_VP_HPP

#include "vptree.hpp"

class DBSCAN_VP : private boost::noncopyable
{
  private:
    static inline double
    dist (const Eigen::VectorXf& p1,
          const Eigen::VectorXf& p2)
    {
      return (p1 - p2).norm ();
    }

    const Dataset::Ptr m_dset;

  public:
    typedef VPTREE<Eigen::VectorXf, dist> TVpTree;
    typedef std::vector<int32_t> Labels;
    typedef boost::shared_ptr<DBSCAN_VP> Ptr;
    int32_t cluster_id;

    DBSCAN_VP (const Dataset::Ptr dset) :
        m_dset (dset),
        cluster_id (0)
    {
    }

    ~DBSCAN_VP ()
    {
    }

    TVpTree::Ptr
    get_vp () const
    {
      return m_vp_tree;
    }

    void
    fit ()
    {
      const Dataset::DataContainer& d = m_dset->data ();

      //const double start = omp_get_wtime ();

      m_vp_tree = boost::make_shared<TVpTree> ();
      m_vp_tree->create (m_dset);

      const size_t dlen = d.size ();

      prepare_labels (dlen);

    }

    const std::vector<double>
    predict_eps (size_t k)
    {
      const Dataset::DataContainer& d = m_dset->data ();

      std::vector<double> r (d.size (), 0.0);

      omp_set_dynamic (1);

#pragma omp parallel for
      for (size_t i = 0; i < d.size (); ++i)
      {
        TVpTree::TNeighborsList nlist;

        m_vp_tree->search_by_k (d[i], k, nlist, true);

        if (nlist.size () >= k)
        {
          r[i] = nlist[0].second;
        }
      }

      std::sort (r.begin (), r.end ());

      return std::move (r);
    }

    uint32_t
    predict (double eps,
             size_t min_elems)
    {

      std::unique_ptr < std::vector<uint32_t> > candidates (new std::vector<uint32_t> ());
      std::unique_ptr < std::vector<uint32_t> > new_candidates (new std::vector<uint32_t> ());

      cluster_id = 0;

      TVpTree::TNeighborsList index_neigh;
      TVpTree::TNeighborsList n_neigh;

      //const double start = omp_get_wtime ();

      const Dataset::DataContainer& d = m_dset->data ();
      const size_t dlen = d.size ();

      for (uint32_t pid = 0; pid < dlen; ++pid)
      {
        if (pid % 10000 == 0)
          continue;
        if (m_labels[pid] >= 0)
          continue;

        find_neighbors (d, eps, pid, index_neigh);

        if (index_neigh.size () < min_elems)
          continue;

        m_labels[pid] = cluster_id;

        candidates->clear ();

        for (const auto& nn : index_neigh)
        {

          if (m_labels[nn.first] >= 0)
            continue;

          m_labels[nn.first] = cluster_id;

          candidates->push_back (nn.first);

        }

        while (candidates->size () > 0)
        {

          new_candidates->clear ();

          //const float csize = float (candidates->size ());

#pragma omp parallel for ordered schedule( dynamic )
          for (size_t j = 0; j < candidates->size (); ++j)
          {

            TVpTree::TNeighborsList c_neigh;
            const uint32_t c_pid = candidates->at (j);

            find_neighbors (d, eps, c_pid, c_neigh);

            if (c_neigh.size () < min_elems)
              continue;
#pragma omp ordered
            {
              for (const auto& nn : c_neigh)
              {

                if (m_labels[nn.first] >= 0)
                  continue;

                m_labels[nn.first] = cluster_id;

                new_candidates->push_back (nn.first);
              }

            }

          }

          std::swap (candidates, new_candidates);
        }
        ++cluster_id;
      }

      return cluster_id;
    }

    void
    reset ()
    {
      m_vp_tree.reset ();
      m_labels.clear ();
    }

    const int32_t
    get_totoal_cluster_number () const
    {
      return cluster_id;
    }

    const Labels&
    get_labels () const
    {
      return m_labels;
    }

  private:
    void
    find_neighbors (const Dataset::DataContainer& d,
                    double eps,
                    uint32_t pid,
                    TVpTree::TNeighborsList& neighbors)
    {
      neighbors.clear ();
      m_vp_tree->search_by_dist (d[pid], eps, neighbors);
    }

    Labels m_labels;

    void
    prepare_labels (size_t s)
    {
      m_labels.resize (s);

      for (auto& l : m_labels)
      {
        l = -1;
      }
    }

    TVpTree::Ptr m_vp_tree;
};

// std::ostream& operator<<( std::ostream& o, DBSCAN& d );

#endif // DBSCAN_VP_HPP
