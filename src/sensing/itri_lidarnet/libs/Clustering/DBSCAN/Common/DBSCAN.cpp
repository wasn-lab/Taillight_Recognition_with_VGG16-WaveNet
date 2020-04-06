#include "DBSCAN.h"

#define UNCLASSIFIED -1
#define NOISE -2

#define SUCCESS 0
#define FAILURE -3

DBSCAN::DBSCAN ()
{
  epsilon = 1;
  minpts = 20;
}

DBSCAN::~DBSCAN ()
{
}

struct point_t
{
    double x, y, z;
    size_t cluster_id;
};

struct node_t
{
    unsigned int index;
    node_t *next;
};

struct epsilon_neighbours_t
{
    unsigned int num_members;
    node_t *head, *tail;
};

node_t *
create_node (unsigned int index)
{
  node_t *n = (node_t *) calloc (1, sizeof(node_t));
  if (n == NULL)
  {
    perror ("[DBSCAN] Failed to allocate node.");
  }
  else
  {
    n->index = index;
    n->next = NULL;
  }
  return n;
}

double
euclidean_dist (point_t *a,
                point_t *b)
{
  return sqrt (pow (a->x - b->x, 2) + pow (a->y - b->y, 2) + pow (a->z - b->z, 2));
}

int
append_at_end (unsigned int index,
               epsilon_neighbours_t *en)
{
  node_t *n = create_node (index);
  if (n == NULL)
  {
    free (en);
    return FAILURE;
  }
  if (en->head == NULL)
  {
    en->head = n;
    en->tail = n;
  }
  else
  {
    en->tail->next = n;
    en->tail = n;
  }
  ++ (en->num_members);
  return SUCCESS;
}

void
destroy_epsilon_neighbours (epsilon_neighbours_t *en)
{
  if (en)
  {
    node_t *t, *h = en->head;
    while (h)
    {
      t = h->next;
      free (h);
      h = t;
    }
    free (en);
  }
}

epsilon_neighbours_t *
get_epsilon_neighbours (unsigned int index,
                        point_t *points,
                        unsigned int num_points,
                        double epsilon)
{
  epsilon_neighbours_t *en = (epsilon_neighbours_t *) calloc (1, sizeof(epsilon_neighbours_t));
  if (en == NULL)
  {
    perror ("Failed to allocate epsilon neighbours.");
    return en;
  }

#pragma omp parallel for
  for (size_t i = 0; i < num_points; ++i)
  {
    if (i != index)
    {
      if (euclidean_dist (&points[index], &points[i]) < epsilon)
      {
#pragma omp critical
        {
          if (append_at_end (i, en) == FAILURE)
          {
            destroy_epsilon_neighbours (en);
            en = NULL;
            i = num_points;
          }
        }
      }
    }
  }
  return en;
}

int
spread (unsigned int index,
        epsilon_neighbours_t *seeds,
        unsigned int cluster_id,
        point_t *points,
        unsigned int num_points,
        double epsilon,
        unsigned int minpts)
{
  epsilon_neighbours_t *spread = get_epsilon_neighbours (index, points, num_points, epsilon);
  if (spread == NULL)
  {
    return FAILURE;
  }
  if (spread->num_members >= minpts)
  {
    node_t *n = spread->head;
    point_t *d;
    while (n)
    {
      d = &points[n->index];
      if ((int)(d->cluster_id) == NOISE || (int)(d->cluster_id) == UNCLASSIFIED)
      {
        if ((int)(d->cluster_id) == UNCLASSIFIED)
        {
          if (append_at_end (n->index, seeds) == FAILURE)
          {
            destroy_epsilon_neighbours (spread);
            return FAILURE;
          }
        }
        d->cluster_id = cluster_id;
      }
      n = n->next;
    }
  }

  destroy_epsilon_neighbours (spread);
  return SUCCESS;
}

void
DBSCAN::setInputCloud (const PointCloud<PointXYZ>::ConstPtr Input)
{
  input = Input;
}
void
DBSCAN::setEpsilon (const double Epsilon)
{
  epsilon = Epsilon;
}
void
DBSCAN::setMinpts (const unsigned int MinPts)
{
  minpts = MinPts;
}
void
DBSCAN::segment (IndicesClusters &clusters)
{
  if (input->size ())
  {
    //input
    point_t *points = (point_t *) calloc (input->size (), sizeof(point_t));

#pragma omp parallel for
    for (size_t i = 0; i < input->size (); i++)
    {
      points[i].x = input->points[i].x;
      points[i].y = input->points[i].y;
      points[i].z = 0;
      points[i].cluster_id = UNCLASSIFIED;
    }

    //process
    unsigned int clusterID = 0;

    for (size_t i = 0; i < input->size (); ++i)
    {
      if ((int)(points[i].cluster_id) == UNCLASSIFIED)
      {
        epsilon_neighbours_t *seeds = get_epsilon_neighbours (i, points, input->size (), epsilon);
        if (seeds != NULL)
        {
          if (seeds->num_members < minpts)
          {
            points[i].cluster_id = NOISE;
          }
          else
          {
            points[i].cluster_id = clusterID;
            node_t *h = seeds->head;

            while (h)
            {
              points[h->index].cluster_id = clusterID;
              h = h->next;
            }

            h = seeds->head;
            while (h)
            {
              spread (h->index, seeds, clusterID, points, input->size (), epsilon, minpts);
              h = h->next;
            }
            // CORE_POINT
            ++clusterID;
          }
          destroy_epsilon_neighbours (seeds);
        }
      }
    }

    //output
    PointIndices buff[clusterID];

#pragma omp parallel for
    for (size_t i = 0; i < input->size (); i++)  //scan all points
    {
      for (size_t j = 0; j < clusterID; j++)  //scan cluster id
      {
        if (points[i].cluster_id == j)
        {
#pragma omp critical
          {
            buff[j].indices.push_back (i);
          }
        }
      }
    }

    for (size_t k = 0; k < clusterID; k++)
    {
      clusters.push_back (buff[k]);
    }

    free (points);
  }
}
