#ifndef DATASET_HPP
#define DATASET_HPP

#include "../../UserDefine.h"

class Dataset : private boost::noncopyable
{
  public:
    typedef boost::shared_ptr<Dataset> Ptr;
    typedef std::vector<Eigen::VectorXf> DataContainer;
    typedef std::vector<size_t> LabelsContainer;

    struct point_t
    {
        double x, y, z;
        int cluster_id;
    };

    static Ptr
    create ()
    {
      return boost::make_shared<Dataset> ();
    }

    Dataset () :
        m_rows (0),
        m_cols (0)
    {
    }

    void
    gen_data (size_t rows,
              size_t cols)
    {
      _data.clear ();
      _labels.resize (rows);

      for (size_t i = 0; i < rows; ++i)
      {
        Eigen::VectorXf col_vector (cols);
        for (size_t j = 0; j < cols; ++j)
        {
          col_vector (j) = (-1.0 + rand () * (2.0) / RAND_MAX);
        }
        _data.emplace_back (col_vector);
      }
    }

    void
    load_pcl (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr input)
    {
      _data.clear ();
      _data.resize (input->size ());
      _labels.clear ();
      _labels.resize (input->size ());
      m_cols = 3;
      m_rows = input->size ();

      Eigen::VectorXf col_vector (3);

      for (size_t i = 0; i < input->size (); ++i)
      {
        col_vector (0) = input->points[i].x;
        col_vector (1) = input->points[i].y;
        col_vector (2) = input->points[i].z;

        _data.at (i) = col_vector;
      }

    }

    void
    load_pcl (const pcl::PointCloud<pcl::PointXYZIL>::ConstPtr input)
    {
      _data.clear ();
      _data.resize (input->size ());
      _labels.clear ();
      _labels.resize (input->size ());
      m_cols = 5;
      m_rows = input->size ();

      Eigen::VectorXf col_vector (5);

      for (size_t i = 0; i < input->size (); ++i)
      {
        col_vector (0) = input->points[i].x;
        col_vector (1) = input->points[i].y;
        col_vector (2) = input->points[i].z;
        col_vector (3) = input->points[i].intensity;
        col_vector (4) = input->points[i].label;

        _data.at (i) = col_vector;
      }

    }

    bool
    load_csv (const std::string& csv_file_path)
    {
      std::ifstream in (csv_file_path);

      if (!in.is_open ())
      {
        return false;
      }

      TKnownLabels known_labels;
      m_everse_labels.clear ();

      std::string line;

      std::vector<float> row_cache;
      _labels.clear ();

      size_t label_idx = 0;

      while (std::getline (in, line))
      {
        if (!line.size ())
        {
          continue;
        }

        row_cache.clear ();

        const char* ptr = line.c_str ();
        size_t len = line.length ();

        const char* start = ptr;
        for (size_t i = 0; i < len; ++i)
        {

          if (ptr[i] == ',')
          {
            row_cache.push_back (std::atof (start));
            start = ptr + i + 1;
          }
        }

        const std::string label_str (start);

        auto r = known_labels.find (start);

        size_t found_label = label_idx;

        if (r == known_labels.end ())
        {
          known_labels.insert (std::make_pair (label_str, label_idx));
          m_everse_labels.insert (std::make_pair (label_idx, label_str));
          ++label_idx;
        }
        else
        {
          found_label = r->second;
        }

        if (!m_cols)
        {
          m_cols = row_cache.size ();
        }
        else
        {

          if (m_cols != row_cache.size ())
          {
            continue;
          }
        }

        Eigen::VectorXf col_vector (row_cache.size ());
        for (size_t i = 0; i < row_cache.size (); ++i)
        {
          col_vector (i) = row_cache[i];
        }

        _data.emplace_back (col_vector);
        _labels.push_back (found_label);

        ++m_rows;
      }

      in.close ();

      assert (_data.size () == _labels.size ());

      return true;
    }

    DataContainer&
    data ()
    {
      return _data;
    }

    const std::string
    get_label (size_t id) const
    {
      auto r = m_everse_labels.find (_labels[id]);
      if (r == m_everse_labels.end ())
      {
        return "Unknown";
      }
      return r->second;
    }

    size_t
    num_points () const
    {
      return m_cols * m_rows;
    }

    size_t
    rows () const
    {
      return m_rows;
    }

    size_t
    cols () const
    {
      return m_cols;
    }

  protected:
    typedef std::unordered_map<std::string, size_t> TKnownLabels;
    typedef std::unordered_map<size_t, std::string> TReverseLabels;

    DataContainer _data;
    LabelsContainer _labels;
    size_t m_rows;
    size_t m_cols;

    TReverseLabels m_everse_labels;
};

#endif // DATASET_HPP
