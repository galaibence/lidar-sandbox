// STL
#include <iostream>
#include <fstream>
#include <chrono>
#include <cfloat>
#include <string>
#include <vector>
#include <list>
#include <queue>
#include <cmath>
#include <memory>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// PCL
#define PCL_NO_PRECOMPILE
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/ndt.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/pca.h>
#include <pcl/filters/approximate_voxel_grid.h>

//Eigen
#include <Eigen/Dense>

// VLP16_project
#include <VCloud.h>
#include <VelodyneStreamer.h>

using namespace std::chrono_literals;


// SPHERICAL GRID PARAMETERS
const int HORIZONTAL_RESOLUTION = 4;
const float DISTANCE_RESOLUTION = 0.2f;
const float GROUND_HEIGHT = 0.3f;
const int CLUTTER_SIZE = 3;

// VOXEL GRID PARAMETERS
const float GRID_RESOLUTION = 0.1f;

// SCAN HOUGH TRANSFORM PARAMETERS
const float RESOLUTION = 0.05f;
const int MAX_D = 2;
const int N = 2 * MAX_D / RESOLUTION;
const float ROT_RESOLUTION = 0.5f;
const int MAX_ROT_D = 5;
const int MANHATTAN = 3;

struct PCLMyPointType {
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D;
  PCL_ADD_RGB

  float Intensity;
  uint16_t SensorAngle;
  uint16_t Distance;
  uchar SensorIntensity;
  char RowNum;

  int _unused;

  int cl;
  int ptIndex;
  int timeStamp;
  int cellNum;
  int id;
  int cluster;
  int learning_class;
  double sensor_x;
  double sensor_y;
  double sensor_z;
  uint PacketID;


  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned

} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PCLMyPointType,
(float, x, x)
(float, y, y)
(float, z, z)
(float, rgb, rgb)
(float, normal_x, normal_x)
(float, normal_y, normal_y)
(float, normal_z, normal_z)
(float, Intensity, Intensity)
(unsigned short int, SensorAngle, SensorAngle)
(unsigned short int, Distance, Distance)
(unsigned char, SensorIntensity, SensorIntensity)
(int, cellNum, cellNum)
(int, id, point_id)
(int, cluster, point_cluster)
(int, learning_class, point_class)
(double, sensor_x, sensor_x)
(double, sensor_y, sensor_y)
(double, sensor_z, sensor_z))



struct StopWatch {
  std::chrono::steady_clock::time_point _start, _end;
  
  void start() {
    _start = std::chrono::steady_clock::now();
  }

  void end() {
    _end = std::chrono::steady_clock::now();
  }

  float duration() {
    return std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count() / 1000.0f;
  }

};

struct Cell {
  int hi, vi, di, label, object_id;

  float min_x, min_y, min_z, max_x, max_y, max_z;

  bool visited;

  std::vector<VPoint*> points;
  std::vector<Cell*> neighbours;

  Cell() :hi{ -1 }, vi{ -1 }, di{ -1 }, label{ -1 }, object_id{ -1 }, min_x{ (float)INT_MAX }, min_y{ (float)INT_MAX }, min_z{ (float)INT_MAX },
    max_x{ (float)INT_MIN }, max_y{ (float)INT_MIN }, max_z{ (float)INT_MIN }, visited{ false } {};

  Cell(int _hi, int _vi, int _di, int _label = -1) 
    :hi{ _hi }, vi{ _vi }, di { _di }, label{ _label }, object_id{ -1 }, visited{ false } {
    min_x = INT_MAX; min_y = INT_MAX; min_z = INT_MAX;
    max_x = INT_MIN; max_y = INT_MIN; max_z = INT_MIN;
  };

  void push_back(VPoint* p) {
    if (p->x < min_x) min_x = p->x;
    if (p->y < min_y) min_y = p->y;
    if (p->z < min_z) min_z = p->z;

    if (p->x > max_x) max_x = p->x;
    if (p->y > max_y) max_y = p->y;
    if (p->z > max_z) max_z = p->z;

    points.push_back(p);
  }
};

struct Object {
  int id;
  float x, y, z;
  std::vector<Cell*> cells;

  Object(int _id) :id{ _id } {}

  void calculateCenterPoint() {
    float minx = INT_MAX, miny = INT_MAX, minz = INT_MAX;
    float maxx = INT_MIN, maxy = INT_MIN, maxz = INT_MIN;
    for (size_t i = 0; i < cells.size(); i++) {
      for (size_t j = 0; j < cells[i]->points.size(); j++) {
        if (cells[i]->points[j]->x < minx) minx = cells[i]->points[j]->x;
        if (cells[i]->points[j]->y < miny) miny = cells[i]->points[j]->y;
        if (cells[i]->points[j]->z < minz) minz = cells[i]->points[j]->z;

        if (cells[i]->points[j]->x > maxx) maxx = cells[i]->points[j]->x;
        if (cells[i]->points[j]->y > maxy) maxy = cells[i]->points[j]->y;
        if (cells[i]->points[j]->z > maxz) maxz = cells[i]->points[j]->z;
      }
    }

    x = (minx + maxx) / 2;
    y = (miny + maxy) / 2;
    z = (minz + maxz) / 2;
  }
};


struct SphericalGrid {
  std::vector<std::vector<std::list<Cell>>> voxels;
  VCloud* cloud;
  SensorType type;

  void operator=(SphericalGrid& other) {
    voxels = other.voxels;
    cloud = other.cloud;
    other.cloud = nullptr;
  }

  SphericalGrid(VCloud* _cloud, SensorType _type) :cloud{ _cloud }, type{ _type } {
    for (int i = 0; i < 360 / HORIZONTAL_RESOLUTION; i++) {
      voxels.push_back(std::vector<std::list<Cell>>((type == SensorType::VLP16 ? 8 : 8 )));
    }
  }

  void fillGrid(bool debug = false) {
    int vi, hi, di;
    int divider = (type == SensorType::VLP16 ? 2 : 8);
    bool insert_before = false;
    for (size_t i = 0; i < cloud->size(); i++) {
      if (cloud->at(i)->distance > 0.5f && cloud->at(i)->distance < 50.0f) {
        //vi = (cloud->at(i)->elevation + 15) / 4;
        vi = (cloud->at(i)->laser_id) / divider;
        
        hi = floor(cloud->at(i)->azimuth / 100 / HORIZONTAL_RESOLUTION);
        di = int((cloud->at(i)->distance - 0.5f) / DISTANCE_RESOLUTION);

        if (voxels[hi][vi].empty()) {
          voxels[hi][vi].push_front(Cell(hi, vi, di));
          voxels[hi][vi].begin()->push_back(cloud->at(i));
        }
        else {
          auto it = voxels[hi][vi].begin();
          for (; it != voxels[hi][vi].end(); it++) {
            if (it->di == di) {
              insert_before = false;
              it->push_back(cloud->at(i));
              break;
            }
            if (it->di > di) {
              insert_before = true;
              break;
            }
          }
          if (insert_before) {
            voxels[hi][vi].insert(it, Cell(hi, vi, di));
            --it;
            it->push_back(cloud->at(i));
          }
        }
      }
    }
  }

  void linkNeighbours() {
    size_t i2, j2;
    int d2;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
        for (auto it = voxels[i][j].begin(); it != voxels[i][j].end(); it++) {
          // FOR EACH CELL TAKE IT'S POSSIBLE NEIGHBOURING INDICES
          for (int n = -1; n < 2; n++) {
            for (int m = -1; m < 2; m++) {
              for (int k = -1; k < 2; k++) {
                // DON'T LINK WITH ITSELF
                if (n == 0 && m == 0 && k == 0) continue;

                /*if (i + n < 0) i2 = voxels.size() - 1;
                else */i2 = (i + n) % voxels.size();   // horizontal connectivity
                if (j + m < 0) continue; /*j2 = voxels[i].size() - 1;*/
                else if ((j + m) >= voxels[i].size()) continue;
                else j2 = (j + m) % voxels[i].size();  // vertical connectivity
                if (it->di + k < 0) continue;  
                else d2 = it->di + k;                       // depth connectivity

                // LINK IF FOUND
                for (auto it2 = voxels[i2][j2].begin(); it2 != voxels[i2][j2].end(); it2++) {
                  if (it2->di == d2) {
                    it->neighbours.push_back(&(*it2));
                  }
                  else if (it2->di > d2) break;
                }
              }
            }
          }
        }
      }
    }
  }

  void clusterCells() {
    float z_max, z_min;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
        for (auto it = voxels[i][j].begin(); it != voxels[i][j].end(); it++) {
          z_max = it->max_z;
          z_min = it->min_z;
          for (auto cell : it->neighbours) {
            if (cell->max_z > z_max) z_max = cell->max_z;
            if (cell->min_z < z_min) z_min = cell->min_z;
          }

          if (it->points.size() < CLUTTER_SIZE) it->label = 1;
          else if (z_max - z_min < GROUND_HEIGHT) it->label = 2;
          else it->label = 3;
        }
      }
    }
  }

  std::vector<Object> defineConnectedPatches() {
    std::vector<Object> objects;

    int object_number = 0;
    std::queue<Cell*> cell_queue;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
        for (auto it = voxels[i][j].begin(); it != voxels[i][j].end(); it++) {
          if (it->visited)
            continue;
          else if (it->label == 3) {
            Object object(object_number);
            it->visited = true;
            cell_queue.push(&(*it));
            while (!cell_queue.empty()) {
              for (size_t n = 0; n < cell_queue.front()->neighbours.size(); n++) {
                if (!cell_queue.front()->neighbours[n]->visited &&
                  cell_queue.front()->neighbours[n]->label == 3)

                  cell_queue.push(cell_queue.front()->neighbours[n]);
                cell_queue.front()->neighbours[n]->visited = true;
              }
              cell_queue.front()->object_id = object_number;
              object.cells.push_back(cell_queue.front());
              cell_queue.pop();
            }
            object_number++;
            object.calculateCenterPoint();
            objects.push_back(object);
          }
        }
      }
    }
    return objects;
  }
};

struct VoxelGrid {
  std::vector<std::vector<std::list<Cell>>> voxels;
  VCloud* cloud;
  SensorType type;

  void operator=(VoxelGrid& other) {
    voxels = other.voxels;
    cloud = other.cloud;
    other.cloud = nullptr;
  }

  VoxelGrid(VCloud* _cloud, SensorType _type) :cloud{ _cloud }, type{ _type } {
    for (int i = 0; i < 50.0f / GRID_RESOLUTION; i++) {
      voxels.push_back(std::vector<std::list<Cell>>(50.0f / GRID_RESOLUTION));
    }
  }

  void fillGrid(bool debug = false) {
    int xi, yi, zi;
    bool insert_before = false;
    for (size_t i = 0; i < cloud->size(); i++) {
      if (fabs(cloud->at(i)->x) >= 0.0f && fabs(cloud->at(i)->y) >= 0.0f &&
         fabs(cloud->at(i)->x) < 25.0f && fabs(cloud->at(i)->y) < 25.0f) {
        //vi = (cloud->at(i)->elevation + 15) / 4;
        xi = (cloud->at(i)->x + 25.0f) / GRID_RESOLUTION;
        yi = (cloud->at(i)->y + 25.0f) / GRID_RESOLUTION;
        zi = cloud->at(i)->z / GRID_RESOLUTION;

        if (voxels[xi][yi].empty()) {
          voxels[xi][yi].push_front(Cell(xi, yi, zi));
          voxels[xi][yi].begin()->push_back(cloud->at(i));
        }
        else {
          auto it = voxels[xi][yi].begin();
          for (; it != voxels[xi][yi].end(); it++) {
            if (it->di == zi) {
              insert_before = false;
              it->push_back(cloud->at(i));
              break;
            }
            if (it->di > zi) {
              insert_before = true;
              break;
            }
          }
          if (insert_before) {
            voxels[xi][yi].insert(it, Cell(xi, yi, zi));
            --it;
            it->push_back(cloud->at(i));
          }
        }
      }
    }
  }

  void linkNeighbours() {
    size_t i2, j2;
    int d2;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
        for (auto it = voxels[i][j].begin(); it != voxels[i][j].end(); it++) {
          // FOR EACH CELL TAKE IT'S POSSIBLE NEIGHBOURING INDICES
          for (int n = -1; n < 2; n++) {
            for (int m = -1; m < 2; m++) {
              for (int k = -1; k < 2; k++) {
                // DON'T LINK WITH ITSELF
                if (n == 0 && m == 0 && k == 0) continue;

                if (i + n < 0) i2 = voxels.size() - 1;
                else i2 = (i + n) % voxels.size();
                if (j + m < 0) j2 = voxels[i].size() - 1;
                else j2 = (j + m) % voxels[i].size();
                if (it->di + k < 0) continue;
                else d2 = it->di + k;

                // LINK IF FOUND
                for (auto it2 = voxels[i2][j2].begin(); it2 != voxels[i2][j2].end(); it2++) {
                  if (it2->di == d2) {
                    it->neighbours.push_back(&(*it2));
                  }
                  else if (it2->di > d2) break;
                }
              }
            }
          }
        }
      }
    }
  }

  void clusterCells() {
    float z_max, z_min;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
        for (auto it = voxels[i][j].begin(); it != voxels[i][j].end(); it++) {
          z_max = it->max_z;
          z_min = it->min_z;
          for (auto cell : it->neighbours) {
            if (cell->max_z > z_max) z_max = cell->max_z;
            if (cell->min_z < z_min) z_min = cell->min_z;
          }

          if (it->points.size() < CLUTTER_SIZE) it->label = 1;
          else if (z_max - z_min < GROUND_HEIGHT) it->label = 2;
          else it->label = 3;
        }
      }
    }
  }

  std::vector<Object> defineConnectedPatches() {
    std::vector<Object> objects;

    int object_number = 0;
    std::queue<Cell*> cell_queue;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
        for (auto it = voxels[i][j].begin(); it != voxels[i][j].end(); it++) {
          if (it->visited)
            continue;
          else if (it->label == 3) {
            Object object(object_number);
            it->visited = true;
            cell_queue.push(&(*it));
            while (!cell_queue.empty()) {
              for (size_t n = 0; n < cell_queue.front()->neighbours.size(); n++) {
                if (!cell_queue.front()->neighbours[n]->visited &&
                  cell_queue.front()->neighbours[n]->label == 3)

                  cell_queue.push(cell_queue.front()->neighbours[n]);
                cell_queue.front()->neighbours[n]->visited = true;
              }
              cell_queue.front()->object_id = object_number;
              object.cells.push_back(cell_queue.front());
              cell_queue.pop();
            }
            object_number++;
            object.calculateCenterPoint();
            objects.push_back(object);
          }
        }
      }
    }
    return objects;
  }
};

struct SimpleGrid {
  std::vector<std::vector<Cell>> voxels;
  VCloud* cloud;
  SensorType type;

  void operator=(SimpleGrid& other) {
    voxels = other.voxels;
    cloud = other.cloud;
    other.cloud = nullptr;
  }

  SimpleGrid(VCloud* _cloud, SensorType _type) :cloud{ _cloud }, type{ _type } {
    for (int i = 0; i < 50 / GRID_RESOLUTION; i++) {
      voxels.push_back(std::vector<Cell>(50 / GRID_RESOLUTION));
    }
  }

  void fillGrid(bool debug = false) {
    int xi, yi;
    for (size_t i = 0; i < cloud->size(); i++) {
      if (fabs(cloud->at(i)->x) >= 0.0f && fabs(cloud->at(i)->y) >= 0.0f &&
        fabs(cloud->at(i)->x) < 25.0f && fabs(cloud->at(i)->y) < 25.0f) {

        xi = (cloud->at(i)->x + 25) / GRID_RESOLUTION;
        yi = (cloud->at(i)->y + 25) / GRID_RESOLUTION;

        voxels[xi][yi].push_back(cloud->at(i));
      }
    }
  }

  void linkNeighbours() {
    size_t i2, j2;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
          for (int n = -1; n < 2; n++) {
            for (int m = -1; m < 2; m++) {
              // DON'T LINK WITH ITSELF
              if (n == 0 && m == 0) continue;

              i2 = i + n;
              j2 = j + m;
              if (i2 < 0 || i2 > voxels.size() - 1 || j2 < 0 || j2 > voxels[i].size() - 1) continue;

              voxels[i][j].neighbours.push_back(&voxels[i2][j2]);
            }
          }
      }
    }
  }

  void clusterCells() {
    float z_max, z_min;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
        z_max = voxels[i][j].max_z;
        z_min = voxels[i][j].min_z;
        for (auto cell : voxels[i][j].neighbours) {
          if (cell->max_z > z_max) z_max = cell->max_z;
          if (cell->min_z < z_min) z_min = cell->min_z;
        }

        if (voxels[i][j].points.size() < CLUTTER_SIZE) voxels[i][j].label = 1;
        else if (z_max - z_min < GROUND_HEIGHT) voxels[i][j].label = 2;
        else voxels[i][j].label = 3;
      }
    }
  }

  std::vector<Object> defineConnectedPatches() {
    std::vector<Object> objects;

    int object_number = 0;
    std::queue<Cell*> cell_queue;
    for (size_t i = 0; i < voxels.size(); i++) {
      for (size_t j = 0; j < voxels[i].size(); j++) {
          if (voxels[i][j].visited)
            continue;
          else if (voxels[i][j].label == 3) {
            Object object(object_number);
            voxels[i][j].visited = true;
            cell_queue.push(&voxels[i][j]);
            while (!cell_queue.empty()) {
              for (size_t n = 0; n < cell_queue.front()->neighbours.size(); n++) {
                if (!cell_queue.front()->neighbours[n]->visited &&
                  cell_queue.front()->neighbours[n]->label == 3)

                  cell_queue.push(cell_queue.front()->neighbours[n]);
                cell_queue.front()->neighbours[n]->visited = true;
              }
              cell_queue.front()->object_id = object_number;
              object.cells.push_back(cell_queue.front());
              cell_queue.pop();
            }
            object_number++;
            object.calculateCenterPoint();
            objects.push_back(object);
        }
      }
    }
    return objects;
  }
};


pcl::PointCloud<PCLMyPointType>::Ptr loadPCDASCI(std::string file_name) {
  std::cout << file_name << std::endl;
  pcl::PointCloud<PCLMyPointType>::Ptr cloud(new pcl::PointCloud<PCLMyPointType>());
  pcl::PCDReader reader;
  reader.read(file_name, *cloud);
  for (auto& p : cloud->points) {
    float temp = p.y;
    p.x = p.x * 0.01f;
    p.y = p.z * 0.01f;
    p.z = temp * 0.01f;
  }
  return cloud;
}


Eigen::Matrix4f coarseAlignment(char* transforms, std::vector<Object>& objects1, std::vector<Object>& objects2) {
  float DX, DY, ROT;
  
  int dmax = 0;
  int imax = -1, jmax = -1;
  float rotmax = -1;

  float x1, y1, x2, y2, dx, dy;
  int dx_i, dy_i, rot_i;

  for (float rot = -MAX_ROT_D; rot < MAX_ROT_D + ROT_RESOLUTION; rot += ROT_RESOLUTION) {
    for (size_t i = 0; i < objects1.size(); i++) {
      x1 = cos(DEG2RAD(rot)) * objects1[i].x
        - sin(DEG2RAD(rot)) * objects1[i].y;
      y1 = sin(DEG2RAD(rot)) * objects1[i].x
        + cos(DEG2RAD(rot)) * objects1[i].y;

      for (size_t j = 0; j < objects2.size(); j++) {
        x2 = objects2[j].x;
        y2 = objects2[j].y;

        dx = x2 - x1;
        dy = y2 - y1;

        dx_i = (dx + MAX_D) * (N / MAX_D / 2);
        dy_i = (dy + MAX_D) * (N / MAX_D / 2);
        rot_i = (rot + MAX_ROT_D) / ROT_RESOLUTION;
        if (dx_i > N - MANHATTAN - 1 || dx_i < MANHATTAN || dy_i > N - MANHATTAN - 1 || dy_i < MANHATTAN
          || rot_i < MANHATTAN || rot_i > (2 * MAX_ROT_D / ROT_RESOLUTION + 1) - MANHATTAN - 1)
          continue;
        for (int i2 = -MANHATTAN; i2 < MANHATTAN + 1; i2++) {
          for (int j2 = -MANHATTAN; j2 < MANHATTAN + 1; j2++) {
            for (int r2 = -MANHATTAN; r2 < MANHATTAN + 1; r2++) {
              if (++transforms[(rot_i + r2) * N * N + (dx_i + i2) * N + dy_i + j2] > dmax) {
                dmax = transforms[(rot_i + r2) * N * N + (dx_i + i2) * N + dy_i + j2];
                imax = dx_i + i2;
                jmax = dy_i + j2;
                rotmax = rot_i + r2;
              }
            }
          }
        }
      }
    }
  }

  if (abs(imax - MAX_D * (N / MAX_D / 2)) < MANHATTAN || abs(jmax - MAX_D * (N / MAX_D / 2)) < MANHATTAN) {
    if (abs(imax - MAX_D * (N / MAX_D / 2)) < MANHATTAN)
      imax = MAX_D * (N / MAX_D / 2);
    if (abs(jmax - MAX_D * (N / MAX_D / 2)) < MANHATTAN) {
      jmax = MAX_D * (N / MAX_D / 2);
    }
    if (fabs(rotmax - MAX_ROT_D / ROT_RESOLUTION) < MANHATTAN) {
      rotmax = MAX_ROT_D / ROT_RESOLUTION;
    }
  }
  else {
    if (fabs(rotmax - MAX_ROT_D / ROT_RESOLUTION) < MANHATTAN - 1) {
      rotmax = MAX_ROT_D / ROT_RESOLUTION;
    }
  }

  DX = (float)imax / (N / MAX_D / 2) - MAX_D;
  DY = (float)jmax / (N / MAX_D / 2) - MAX_D;
  ROT = (float)rotmax  * ROT_RESOLUTION - MAX_ROT_D;

  std::cout << "dmax: " << dmax << " DX: " << DX << " DY: " << DY << " ROT: " << ROT << std::endl;
  std::cout << "\t" << imax << " " << jmax << " " << rotmax << std::endl;
  std::cout << std::endl;

  Eigen::Matrix4f post_transform = Eigen::Matrix4f::Identity();
  post_transform(0, 0) = cos(DEG2RAD(ROT));
  post_transform(0, 1) = -sin(DEG2RAD(ROT));
  post_transform(1, 0) = sin(DEG2RAD(ROT));
  post_transform(1, 1) = cos(DEG2RAD(ROT));
  post_transform(0, 3) = DX;
  post_transform(1, 3) = DY;

  return post_transform;
}

Eigen::Matrix4f fineAlignmentNDT(pcl::PointCloud<pcl::PointXYZRGB>::Ptr source, pcl::PointCloud<pcl::PointXYZRGB>::Ptr target) {
  pcl::NormalDistributionsTransform<pcl::PointXYZRGB, pcl::PointXYZRGB> ndt;
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_source(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_target(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ApproximateVoxelGrid<pcl::PointXYZRGB> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize(0.1, 0.1, 0.1);
  approximate_voxel_filter.setInputCloud(source);
  approximate_voxel_filter.filter(*filtered_source);
  approximate_voxel_filter.setInputCloud(target);
  approximate_voxel_filter.filter(*filtered_target);

  ndt.setTransformationEpsilon(0.01);
  ndt.setStepSize(0.1);
  ndt.setResolution(2.0);
  ndt.setMaximumIterations(35);
  
  ndt.setInputSource(filtered_source);
  ndt.setInputTarget(filtered_target);
  
  pcl::PointCloud<pcl::PointXYZRGB> final;
  ndt.align(final);

  return ndt.getFinalTransformation();
}


void iteration(VCloud& cloud0, VCloud& cloud1, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloudSUM, SensorType type, int n, int idx, char* transforms, Eigen::Matrix4f& pretransform, pcl::visualization::CloudViewer& viewer, std::vector<int>& histogram) {
    
  StopWatch sw2, sw;
  sw2.start();

  std::vector<Object> objects0;
  std::vector<Object> objects1;

  sw.start();
  SphericalGrid grid0(&cloud0, type);
  grid0.fillGrid();
  grid0.linkNeighbours();
  grid0.clusterCells();
  objects0 = grid0.defineConnectedPatches();
  sw.end();
  //std::cout << "0: " << sw.duration() << " ms. " << objects0.size() << std::endl;
  
  pcl::PointXYZRGB pclp;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud0(new pcl::PointCloud<pcl::PointXYZRGB>());
  for (size_t i = 0; i < cloud0.size(); i++) {
    pclp.x = cloud0.at(i)->x;
    pclp.y = cloud0.at(i)->y;
    pclp.z = cloud0.at(i)->z;
    //pclp.intensity = cloud0.operator[](i).intensity;
    pclcloud0->push_back(pclp);
    if (idx == 1)
      pclcloudSUM->push_back(pclp);
  }

  pcl::PointCloud<pcl::PointXY>::Ptr xycloud(new pcl::PointCloud<pcl::PointXY>());
  pcl::PointXY xypoint;
  for (auto object : objects0) {
    xypoint.x = object.x;
    xypoint.y = object.y;
    xycloud->push_back(xypoint);
  }

  sw.start();
  SphericalGrid grid1(&cloud1, type);
  grid1.fillGrid();
  grid1.linkNeighbours();
  grid1.clusterCells();
  objects1 = grid1.defineConnectedPatches();
  sw.end();
  //std::cout << "1: " << sw.duration() << " ms. " << objects1.size() << std::endl;

  sw.start();
  Eigen::Matrix4f transform1 = Eigen::Matrix4f::Identity(); //coarseAlignment(transforms, objects1, objects0);
  Eigen::Matrix4f transform2 = Eigen::Matrix4f::Identity(); //fineAlignmentNDT(pclcloud1, pclcloud0); //
  Eigen::Matrix4f transform = pretransform * transform1 * transform2;
  sw.end();
  //std::cout << "2: " << sw.duration() << " ms." << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud1(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointXYZRGB prgb;
  for (size_t i = 0; i < cloud1.size(); i++) {
    prgb.x = cloud1[i].x;
    prgb.y = cloud1[i].y;
    prgb.z = cloud1[i].z;
    //prgb.r = 0;
    //prgb.g = 0;
    //prgb.b = 0;
    pclcloud1->push_back(prgb);
  }

  pcl::PointCloud<PCLMyPointType> mycloud;

#ifdef COLOR_CLUSTERS
  for (auto cell_list_vector : grid1.voxels) {
    for (auto cell_list : cell_list_vector) {
      for (auto cell : cell_list) {
        if (cell.label == 3) continue;
        prgb.r = cell.label == 2 ? 255 : 0;
        prgb.g = cell.label == 2 ? 0 : 255;
        prgb.b = cell.label == 2 ? 0 : 0;
        for (auto point : cell.points) {
          prgb.x = point->x;
          prgb.y = point->y;
          prgb.z = point->z;
          pclcloud1->push_back(prgb);
        }
      }
    }
  }
#else
  for (auto object : objects1) {
    //pclcloud1->push_back(prgb);
    ////float min_z = object.cells[0]->points[0]->z;
    ////float max_z = object.cells[0]->points[0]->z;
    ////for (auto cell : object.cells) {
    ////  for (auto point : cell->points) {
    ////    if (point->z < min_z) min_z = point->z;
    ////    if (point->z > max_z) max_z = point->z;
    ////  }
    ////}
    //if (max_z - min_z > 2.5) { // magas objektum
    //  mypoint.r = 255;
    //  mypoint.g = 255;
    //  mypoint.b = 0;
    //}
    //else { // alacsony objektum
    //  mypoint.r = 0;
    //  mypoint.g = 255;
    //  mypoint.b = 255;
    //}

    prgb.r = rand() % 255;
    prgb.g = rand() % 255;
    prgb.b = rand() % 255;
    for (auto cell : object.cells)
      for (auto point : cell->points) {
        prgb.x = point->x;
        prgb.y = point->y;
        prgb.z = point->z;
        pclcloud1->push_back(prgb);
      }
  }
#endif

  //std::string SZTAKI(std::getenv("SZTAKIPATH"));
  //CreateDirectory((SZTAKI + "\\DATA\\PCD\\STREAM\\HDL64\\" + std::to_string(n)).c_str(), nullptr);
  //pcl::io::savePCDFileBinary(SZTAKI + "\\DATA\\PCD\\STREAM\\HDL64\\" + std::to_string(n) + "\\" + std::to_string(idx) + ".pcd", mycloud);

  //idx = 1;
  // X HISTOGRAM PHASE
  //for (int i = 0; i < cloud0.size(); i++) {
  //  float x = cloud0[i].x * cosf(-18.25f * M_PI / 180.0f) - cloud0[i].y * sinf(-18.25f * M_PI / 180.0f);
  //  float y = cloud0[i].x * sinf(-18.25f * M_PI / 180.0f) + cloud0[i].y * cosf(-18.25f * M_PI / 180.0f);
  //  //    cloud0.at(i)->x = x;
  //  //    cloud0.at(i)->y = y;
  //}
  float min_x = 0.0f, min_y = 0.0f, max_x = 0.0f, max_y = 0.0f;
  int hist_max = 0;
  std::vector<int> x_histogram(200 * 10);
  //for (auto object : objects1)
    //for (auto cell : object.cells)
  for (size_t i = 0; i < cloud0.size(); i++) {
    VPoint point = cloud0[i];
        float x = point.x * cosf(-18.25f * M_PI / 180.0f) - point.y * sinf(-18.25f * M_PI / 180.0f);
        float y = point.x * sinf(-18.25f * M_PI / 180.0f) + point.y * cosf(-18.25f * M_PI / 180.0f);
        //    cloud1.at(i)->x = x;
        //    cloud1.at(i)->y = y;
        if ((int)(x * 10 + 1000) >= 0 && (int)(x * 10 + 1000) < 2000 && point.z > 0 && point.distance > 1.5f) {
          if (++x_histogram[(int)(x * 10 + 1000)] > hist_max) {
            hist_max = x_histogram[(int)(x * 10 + 1000)];
          }
        }
        if (x < min_x) min_x = x;
        if (y < min_y) min_y = y;
        if (x > max_x) max_x = x;
        if (y > max_y) max_y = y;
      }
  hist_max = 0;
  for (int i = 0; i < 2000; i++) {
    histogram[i] = (x_histogram[i] + idx * histogram[i]) / (idx + 1);
    if (histogram[i] > hist_max) hist_max = histogram[i];
  }
  cv::Mat x_hist = cv::Mat::zeros(hist_max, 2000, 0);
  for (size_t i = 0; i < histogram.size(); i++) {
    for (int j = 0; j < histogram[i]; j++) {
      if (j > x_hist.cols) continue;
      x_hist.at<uchar>(j, i) = 255;
    }
  }
  cv::resize(x_hist, x_hist, cv::Size(1000, 500));
  cv::imshow("X histogram", x_hist);
  cv::waitKey(1);
  // ~X HISTOGRAM PHASE

  pclcloud0->clear();
  pclcloud0->resize(0);
  for (size_t i = 0; i < cloud0.size(); i++) {
    prgb.x = cloud0.at(i)->x;
    prgb.y = cloud0.at(i)->y;
    prgb.z = cloud0.at(i)->z;
    pclcloud0->push_back(prgb);
  }


  std::string stype = type == SensorType::VLP16 ? "VLP" : "HDL";
  //pcl::transformPointCloud(*pclcloud1, *pclcloud1, transform);
  //if (idx % 5 == 0) {
  //  pcl::io::savePCDFileBinary("C:\\Users\\Bence\\Desktop\\SZTAKI\\DATA\\PCD\\OUTPUT\\" + stype + "\\" 
  //  + std::to_string(n) + "_15_" + std::to_string(idx - 15 + 14) + ".pcd", *pclcloudSUM);
  //  transform = Eigen::Matrix4f::Identity();
  //  pclcloudSUM->clear();
  //  pclcloudSUM->resize(0);
  //}

  for (auto p : pclcloud1->points) {
    pclcloudSUM->push_back(p);
  }

  pretransform = transform;

  //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud12(new pcl::PointCloud<pcl::PointXYZI>());
  //for (auto p : pclcloud1->points) {
  //  cloud12->push_back(p);
  //}
  //for (auto p : pclcloud0->points) {
  //  cloud12->push_back(p);
  //}
  viewer.showCloud(pclcloud1);

  cloud0 = cloud1;

  for (int k = 0; k < (2 * MAX_ROT_D / ROT_RESOLUTION + 1) * N * N; k++) {
    transforms[k] = 0;
  }

  sw2.end();
  //cv::Mat range_image = cv::Mat::zeros(type == VLP16 ? 16 : 64, 360 * 3, CV_8UC1);
  //cv::Mat object_image = cv::Mat::zeros(type == VLP16 ? 16 : 64, 360 * 3, CV_8UC3);
  //cv::Mat intensity_image = cv::Mat::zeros(type == VLP16 ? 16 : 64, 360 * 3, CV_8UC3);
  //cv::Vec3b color;
  //for (auto k = 0; k < cloud1.size(); k++) {
  //  if (cloud1.at(k)->distance >= 50 || cloud1.at(k)->distance < 0.5f) continue;
  //  range_image.at<uchar>((type == VLP16 ? 15 : 63) - cloud1.at(k)->laser_id,
  //    ((cloud1.at(k)->azimuth / 33) + 180 * 3) % (360 * 3)) =
  //    (unsigned char)(255 - (cloud1.at(k)->distance - 0.5f) / (50.0f - 0.5f) * 255);
  //  int I = cloud1.at(k)->intensity * 2;
  //  color(0) = I > 255 ? 255 : 0;
  //  color(1) = I > 255 ? 255 - (I - 255) : I;
  //  color(2) = I > 255 ? 0 : 255;
  //  intensity_image.at<cv::Vec3b>((type == VLP16 ? 15 : 63) - cloud1.at(k)->laser_id,
  //    ((cloud1.at(k)->azimuth / 33) + 180 * 3) % (360 * 3)) = color;
  //}
  //for (int i = 0; i < object_image.rows; i++) {
  //  for (int j = 0; j < object_image.cols; j++) {
  //    object_image.at<cv::Vec3b>(i, j) =
  //      cv::Vec3b(
  //        range_image.at<uchar>(i, j),
  //        range_image.at<uchar>(i, j),
  //        range_image.at<uchar>(i, j)
  //      );
  //  }
  //}
  //for (auto object : objects1) {
  //  int r = 255;
  //  int g = 0;
  //  int b = 0;
  //  color = cv::Vec3b(r, g, b);
  //  for (auto cell : object.cells) {
  //    for (auto point : cell->points) {
  //      object_image.at<cv::Vec3b>((type == VLP16 ? 15 : 63) - point->laser_id,
  //        ((point->azimuth / 33) + 180 * 3) % (360 * 3)) = color;
  //    }
  //  }
  //}
  //cv::medianBlur(range_image, range_image, 3);
  //cv::medianBlur(intensity_image, intensity_image, 3);
  //cv::medianBlur(object_image, object_image, 3);
  //cv::resize(range_image, range_image, cv::Size(range_image.cols, range_image.rows * 4));
  //cv::resize(intensity_image, intensity_image, cv::Size(intensity_image.cols, intensity_image.rows * 4));
  //cv::resize(object_image, object_image, cv::Size(object_image.cols, object_image.rows * 4));
  //cv::imshow("range_image", range_image);
  //cv::imshow("object_image", object_image);
  //cv::imshow("intensity_image", intensity_image);
  ////cv::imwrite("C:\\Users\\bence\\Desktop\\images\\vlp_" + std::to_string(n) + "_R" + std::to_string(idx) + ".jpg", range_image);
  ////cv::imwrite("C:\\Users\\bence\\Desktop\\images\\vlp_" + std::to_string(n) + "_I" + std::to_string(idx) + ".jpg", intensity_image);
  //cv::waitKey(1);
}

void iteration_pcl(pcl::PointCloud<PCLMyPointType>::Ptr& inputcloud0, 
                   pcl::PointCloud<PCLMyPointType>::Ptr& inputcloud1, 
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloudSUM,
                   SensorType type, 
                   int idx, char* transforms, 
                   Eigen::Matrix4f& pretransform, 
                   pcl::visualization::CloudViewer& viewer, 
                   std::vector<int>& histogram) {

  StopWatch sw2, sw;
  sw2.start();

  std::vector<Object> objects0;
  std::vector<Object> objects1;

  VPoint vp;
  VCloud cloud0, cloud1;

  std::cout << cloud0.size() << " " << cloud1.size() << std::endl;

  for (auto p : inputcloud0->points) {
    vp.x = p.x;
    vp.y = p.y;
    vp.z = p.z;
    cloud0.push_back(vp);
  }
  for (auto p : inputcloud1->points) {
    vp.x = p.x;
    vp.y = p.y;
    vp.z = p.z;
    cloud1.push_back(vp);
  }

  sw.start();
  SimpleGrid grid0(&cloud0, type);
  grid0.fillGrid();
  grid0.linkNeighbours();
  grid0.clusterCells();
  objects0 = grid0.defineConnectedPatches();
  sw.end();
  std::cout << "0: " << sw.duration() << " ms. " << objects0.size() << std::endl;

  pcl::PointXYZRGB pclp;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud0(new pcl::PointCloud<pcl::PointXYZRGB>());
  for (size_t i = 0; i < cloud0.size(); i++) {
    pclp.x = cloud0.at(i)->x;
    pclp.y = cloud0.at(i)->y;
    pclp.z = cloud0.at(i)->z;
    //pclp.intensity = cloud0.operator[](i).intensity;
    pclcloud0->push_back(pclp);
    if (idx == 1)
      pclcloudSUM->push_back(pclp);
  }

  sw.start();
  SimpleGrid grid1(&cloud1, type);
  grid1.fillGrid();
  grid1.linkNeighbours();
  grid1.clusterCells();
  objects1 = grid1.defineConnectedPatches();
  sw.end();
  std::cout << "1: " << sw.duration() << " ms. " << objects1.size() << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud1(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointXYZRGB prgb;
  for (size_t i = 0; i < cloud1.size(); i++) {
    prgb.x = cloud1[i].x;
    prgb.y = cloud1[i].y;
    prgb.z = cloud1[i].z;
    //prgb.r = inputcloud1->at(i).r;
    //prgb.g = inputcloud1->at(i).g;
    //prgb.b = inputcloud1->at(i).b;
    pclcloud1->push_back(prgb);
  }


#ifndef COLOR_CLUSTERS
  for (auto cell_list_vector : grid1.voxels) {
    //for (auto cell_list : cell_list_vector)
      for (auto cell : cell_list_vector) {
        //prgb.r = (cell.max_z - cell.min_z) / 3.0f * 255;
        //prgb.g = (cell.max_z - cell.min_z) / 3.0f * 255;
        //prgb.b = (cell.max_z - cell.min_z) / 3.0f * 255;
        prgb.r = cell.label == 2 ? 255 : 0;
        prgb.g = cell.label == 1 ? 255 : 0;
        prgb.b = cell.label == 3 ? 255 : 0;
        for (auto point : cell.points) {
          prgb.x = point->x;
          prgb.y = point->y;
          prgb.z = point->z;
          pclcloud1->push_back(prgb);
        }
      }
  }
#else
  for (auto object : objects1) {
    pclcloud1->push_back(prgb);
    float min_z = object.cells[0]->points[0]->z;
    float max_z = object.cells[0]->points[0]->z;
    for (auto cell : object.cells) {
      for (auto point : cell->points) {
        if (point->z < min_z) min_z = point->z;
        if (point->z > max_z) max_z = point->z;
      }
    }
    if (max_z - min_z > 2.5) { // magas objektum
      prgb.r = 255;
      prgb.g = 255;
      prgb.b = 0;
    }
    else { // alacsony objektum
      prgb.r = 0;
      prgb.g = 255;
      prgb.b = 255;
    }
    for (auto cell : object.cells)
      for (auto point : cell->points) {
        prgb.x = point->x;
        prgb.y = point->y;
        prgb.z = point->z;
        pclcloud1->push_back(prgb);
      }
  }
#endif

  // X HISTOGRAM PHASE
  //for (int i = 0; i < cloud0.size(); i++) {
  //  float x = cloud0[i].x * cosf(-18.25f * M_PI / 180.0f) - cloud0[i].y * sinf(-18.25f * M_PI / 180.0f);
  //  float y = cloud0[i].x * sinf(-18.25f * M_PI / 180.0f) + cloud0[i].y * cosf(-18.25f * M_PI / 180.0f);
  //  //    cloud0.at(i)->x = x;
  //  //    cloud0.at(i)->y = y;
  //}
  //float min_x = 0.0f, min_y = 0.0f, max_x = 0.0f, max_y = 0.0f;
  //int hist_max = 0;
  //std::vector<int> x_histogram(200 * 10);
  //for (auto object : objects1)
  //  for (auto cell : object.cells)
  //    for (auto point : cell->points) {
  //      float x = point->x * cosf(-18.25f * M_PI / 180.0f) - point->y * sinf(-18.25f * M_PI / 180.0f);
  //      float y = point->x * sinf(-18.25f * M_PI / 180.0f) + point->y * cosf(-18.25f * M_PI / 180.0f);
  //      //    cloud1.at(i)->x = x;
  //      //    cloud1.at(i)->y = y;
  //      if ((int)(x * 10 + 1000) >= 0 && (int)(x * 10 + 1000) < 2000 && point->z > 0 && point->distance > 1.5f) {
  //        if (++x_histogram[(int)(x * 10 + 1000)] > hist_max) {
  //          hist_max = x_histogram[(int)(x * 10 + 1000)];
  //        }
  //      }
  //      if (x < min_x) min_x = x;
  //      if (y < min_y) min_y = y;
  //      if (x > max_x) max_x = x;
  //      if (y > max_y) max_y = y;
  //    }
  //hist_max = 0;
  //for (int i = 0; i < 2000; i++) {
  //  histogram[i] = (x_histogram[i] + idx * histogram[i]) / (idx + 1);
  //  if (histogram[i] > hist_max) hist_max = histogram[i];
  //}
  //cv::Mat x_hist = cv::Mat::zeros(hist_max, 2000, 0);
  //for (int i = 0; i < histogram.size(); i++) {
  //  for (int j = 0; j < histogram[i]; j++) {
  //    if (j > x_hist.cols) continue;
  //    x_hist.at<uchar>(j, i) = 255;
  //  }
  //}
  //cv::resize(x_hist, x_hist, cv::Size(1000, 500));
  //cv::imshow("X histogram", x_hist);
  // ~X HISTOGRAM PHASE



  pclcloud0->clear();
  pclcloud0->resize(0);
  for (size_t i = 0; i < cloud0.size(); i++) {
    prgb.x = cloud0.at(i)->x;
    prgb.y = cloud0.at(i)->y;
    prgb.z = cloud0.at(i)->z;
    pclcloud0->push_back(prgb);
  }

  sw.start();
  Eigen::Matrix4f transform1 = coarseAlignment(transforms, objects1, objects0);
  Eigen::Matrix4f transform2 =  Eigen::Matrix4f::Identity(); //fineAlignmentNDT(pclcloud1, pclcloud0); //
  Eigen::Matrix4f transform = pretransform * transform1 * transform2;
  sw.end();
  std::cout << "2: " << sw.duration() << " ms." << std::endl;

  std::string stype = type == SensorType::VLP16 ? "VLP" : "HDL";
  pcl::transformPointCloud(*pclcloud1, *pclcloud1, transform);

  for (auto p : pclcloud1->points) {
    pclcloudSUM->push_back(p);
  }

  if (idx % 5 == 0) {
    //pcl::io::savePCDFileBinary("C:\\Users\\Bence\\Desktop\\OUTPUT\\" + stype + "\\" 
    //+ std::to_string(n) + "_" + std::to_string(idx - 5 + 4) + ".pcd", *pclcloudSUM);
    pcl::io::savePCDFileBinary("C:\\Users\\bence\\Desktop\\New folder\\cloud" + std::to_string(idx) + ".pcd", *pclcloudSUM);

    transform = Eigen::Matrix4f::Identity();
    pclcloudSUM->clear();
    pclcloudSUM->resize(0);
  }

  pretransform = transform;

  pcl::PointXYZRGB rgbpoint;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloudrgb(new pcl::PointCloud<pcl::PointXYZRGB>());
  std::vector<int> ids;
  for (auto p : inputcloud1->points) {
    rgbpoint.x = p.x;
    rgbpoint.y = p.y;
    rgbpoint.z = p.z;
    rgbpoint.r = p.id % 255;
    rgbpoint.g = p.id % 255;
    rgbpoint.b = p.id % 255;
    pclcloudrgb->push_back(rgbpoint);
  }
  viewer.showCloud(pclcloud1);

  cloud0 = cloud1;

  for (int k = 0; k < (2 * MAX_ROT_D / ROT_RESOLUTION + 1) * N * N; k++) {
    transforms[k] = 0;
  }

  sw2.end();

  cv::waitKey(1);
}


float vpoint_distance(VPoint& a, VPoint& b) {
  return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

void iteration_segment(VCloud& cloud, pcl::visualization::CloudViewer& viewer, std::vector<cv::Vec3b>& rgb) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  std::vector<int> labels(cloud.size());
  std::vector<VPoint> points;
  std::vector<VPoint> points0;
  pcl::PointXYZRGB pclrgb;

  points.clear();
  points.resize(0);
  for (int j = 0; j < 16; j++) {
    points.push_back(cloud[j]);
  }
  std::sort(points.begin(), points.end(), [](VPoint p1, VPoint p2) { return p1.elevation < p2.elevation; });

  float threshold_vertical = 0.7f;
  float threshold_horizontal = 0.3f;
  int id = 0;
  labels[0] = id++;
  for (int j = 1; j < 16; j++) {
    if (fabs(points[j].distance - points[j - 1].distance) < threshold_vertical) {
      labels[j] = labels[j - 1];
    }
    else {
      labels[j] = id++;
    }
  }

  for (int j = 0; j < 16; j++) {
    pclrgb.x = points[j].x;
    pclrgb.y = points[j].y;
    pclrgb.z = points[j].z;
    pclrgb.r = rgb[labels[j]](0);
    pclrgb.g = rgb[labels[j]](1);
    pclrgb.b = rgb[labels[j]](2);
    pclcloud->push_back(pclrgb);
  }
  points0 = points;

  for (size_t i = 1; i < cloud.size() / 16; i++) {
    points.clear();
    points.resize(0);
    for (int j = 0; j < 16; j++) {
      points.push_back(cloud[i * 16 + j]);
    }
    
    std::sort(points.begin(), points.end(), [](VPoint p1, VPoint p2) { return p1.elevation < p2.elevation; });

    if (vpoint_distance(points[0], points0[0]) < threshold_horizontal) {
      labels[i * 16 + 0] = labels[(i - 1) * 16 + 0];
    }
    else {
      labels[i * 16 + 0] = id++;
    }
    //std::cout << "iteration begin" << std::endl;
    for (int j = 1; j < 16; j++) {
      //std::cout << points[j].distance << std::endl;
      //std::cout << points[j - 1].distance << std::endl;
      if (vpoint_distance(points[j], points0[j]) < threshold_horizontal) {
        labels[i * 16 + j] = labels[(i - 1) * 16 + j];
      }
      else if (vpoint_distance(points[j], points[j - 1]) < threshold_vertical) {
        labels[i * 16 + j] = labels[i * 16 + j - 1];
      }
      else {
        labels[i * 16 + j] = id++;
      }
    }

    for (int j = 0; j < 16; j++) {
      pclrgb.x = points[j].x;
      pclrgb.y = points[j].y;
      pclrgb.z = points[j].z;
      //pclrgb.r = rgb[labels[i * 16 + j]](0);
      //pclrgb.g = rgb[labels[i * 16 + j]](1);
      //pclrgb.b = rgb[labels[i * 16 + j]](2);
      pclcloud->push_back(pclrgb);
    }

    points0 = points;
  }

  std::vector<cv::Vec3b> rgbs(id);
  for (int i = 0; i < id; i++) {
    rgbs[i] = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
  }
  std::cout << id << " " << pclcloud->size() << " " << labels.size() << std::endl;
  for (size_t i = 0; i < pclcloud->size(); i++) {
    pclcloud->at(i).r = rgbs[labels[i]](0);
    pclcloud->at(i).g = rgbs[labels[i]](1);
    pclcloud->at(i).b = rgbs[labels[i]](2);
  }
  viewer.showCloud(pclcloud);
}

void pcap_streamer2(std::string pcap) {
  char* transforms = new char[(int)(2 * MAX_ROT_D / ROT_RESOLUTION + 1) * N * N];
  for (int k = 0; k < (2 * MAX_ROT_D / ROT_RESOLUTION + 1) * N * N; k++) {
    transforms[k] = 0;
  }

  Eigen::Matrix4f pretransform = Eigen::Matrix4f::Identity();

  std::string vlps[12] = {
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-36-16_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-38-53_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-40-24_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
    "/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160202/2016-02-02-09-30-17_Velodyne-VLP-16-Data.pcap",
  };

  std::string hdls[12] = {
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-37-19_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-38-43_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-40-17_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
    "/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-57-03_Velodyne-HDL-Data.pcap",
  };
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud0(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud1(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloudSUM(new pcl::PointCloud<pcl::PointXYZRGB>());
  VCloud cloud0, cloud1;

  pcl::PointXYZI pclp;

  pcl::visualization::CloudViewer viewer("Stream");
  VelodyneStreamer vs;
  SensorType type = SensorType::HDL64;

  //cv::namedWindow("range_image", cv::WINDOW_NORMAL);
  //cv::namedWindow("object_image", cv::WINDOW_NORMAL);
  //cv::namedWindow("intensity_image", cv::WINDOW_NORMAL);
  //cv::Mat range_image = cv::Mat::zeros(type == VLP16 ? 16 : 64, 360 * 3, CV_8UC1);
  //cv::Mat object_image = cv::Mat::zeros(type == VLP16 ? 16 : 64, 360 * 3, CV_8UC3);
  //cv::Mat intensity_image = cv::Mat::zeros(type == VLP16 ? 16 : 64, 360 * 3, CV_8UC3);

  while (!viewer.wasStopped()) {
    for (int n = 2; n < 3; n++) {
      StopWatch sw;
      StopWatch sw2;
      std::cout << hdls[n];
      // vs.open(type == VLP16 ? vlps[n] : hdls[n]);
      vs.open("/media/bence/Data/DATA/PCAP/VLP16/Udvari kukás/2016-08-10-11-27-54_Velodyne-VLP-16-Data.pcap");

      vs.nextFrame(cloud0);

      //std::vector<cv::Vec3b> rgb;
      //for (int i = 0; i < 16; i++) {
      //  rgb.push_back(cv::Vec3b(rand() % 255, rand() % 255, rand() % 255));
      //}
      //for (auto k = 0; k < cloud0.size(); k++) {
      //  if (cloud0.at(k)->distance >= 50.0 || cloud0.at(k)->distance < 0.5f) continue;
      //  range_image.at<uchar>((type == VLP16 ? 15 : 63) - cloud0.at(k)->laser_id,
      //    ((cloud0.at(k)->azimuth / 33) + 180 * 3) % (360 * 3)) =
      //    (unsigned char)(255 - (cloud0.at(k)->distance - 0.5f) / (50.0 - 0.5f) * 255);
      //}
      
      std::vector<int> histogram(2000);

      int idx = 0;
      while (vs.nextFrame(cloud1)) {
        //if (idx > 600 && idx < 800)
          iteration(cloud0, cloud1, pclcloudSUM, type, n, idx, transforms, pretransform, viewer, histogram);
        cloud0 = cloud1;
        idx++;
        //iteration_segment(cloud1, viewer, rgb);
      }

    }
    break;
  }

  delete[] transforms;
}

void pcap_streamer3(std::string pcap) {
  char* transforms = new char[(int)(2 * MAX_ROT_D / ROT_RESOLUTION + 1) * N * N];
  for (int k = 0; k < (2 * MAX_ROT_D / ROT_RESOLUTION + 1) * N * N; k++) {
    transforms[k] = 0;
  }

  Eigen::Matrix4f pretransform = Eigen::Matrix4f::Identity();

  std::string folder = "C:\\Users\\bence\\Desktop\\bartok_id_class_objects_point_clouds\\0-45\\CapturedFrame_0";
  int offset = 0;

  pcl::PointCloud<PCLMyPointType>::Ptr pclcloud0(new pcl::PointCloud<PCLMyPointType>());
  pcl::PointCloud<PCLMyPointType>::Ptr pclcloud1(new pcl::PointCloud<PCLMyPointType>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloudSUM(new pcl::PointCloud<pcl::PointXYZRGB>());

  pcl::PCDReader reader;

  pcl::visualization::CloudViewer viewer("Stream");

  while (!viewer.wasStopped()) {
    StopWatch sw;
    StopWatch sw2;
    SensorType type = SensorType::VLP16;

    pclcloud0 = loadPCDASCI(folder + "0" + std::to_string(offset) + ".pcd_BIN.pcd");

    std::vector<cv::Vec3b> rgb;
    for (int i = 0; i < 16; i++) {
      rgb.push_back(cv::Vec3b(rand() % 255, rand() % 255, rand() % 255));
    }

    std::vector<int> histogram(2000);

    int idx = 1;
    while (idx < 22) {
      std::string lowerThenTen = idx < 10 ? "0" : "";
      pclcloud1 = loadPCDASCI(folder + lowerThenTen + std::to_string(offset + idx) + ".pcd_BIN.pcd");

      iteration_pcl(pclcloud0, pclcloud1, pclcloudSUM, type, idx, transforms, pretransform, viewer, histogram);
      pclcloud0->clear();
      pclcloud0->resize(0);
      for (auto p : pclcloud1->points) {
        pclcloud0->push_back(p);
      }

      idx++;
    }
    break;
  }

  delete[] transforms;
}


std::vector<std::vector<int>> hdl32_segmentation(std::vector<std::vector<VPoint>>& pointlist) {
  std::vector<std::vector<int>> labels(pointlist.size());

  for (size_t i = 1; i < pointlist.size() - 1; i++) {
    for (size_t j = 1; j < pointlist[i].size() - 1; j++) {

    }
  }

  return labels;
}


struct P2 {
  float x, y;
  P2(float _x, float _y, float _z) :x{sqrtf(_x*_x + _y*_y)}, y{_z} {}
};

void graphCanny(std::vector<std::vector<VPoint>>& pointlist, pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
  pcl::PointXYZRGB pclp;

  auto VPdist = [&](VPoint p1, VPoint p2) { 
    return sqrtf((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z)); 
  };

  auto f = [&](P2 p1, P2 p2, P2 p3, P2 p4) {
    float A = p3.y - p2.y;
    float B = -p3.x + p2.x;
    float C = A*p2.x + B*p2.y;

    return (fabs(p1.x * A + p1.y * B + C) + fabs(p4.x * A + p4.y * B + C)) / (2 * sqrtf(A*A + B*B));
  };

  for (size_t i = 1; i < pointlist.size() - 1; i++) {
    for (size_t j = 2; j < pointlist[i].size() - 2; j++) {
      pclp.x = pointlist[i][j].x;
      pclp.y = pointlist[i][j].y;
      pclp.z = pointlist[i][j].z;

      float f_value = f({pointlist[i][j - 1].x, pointlist[i][j - 1].y, pointlist[i][j - 1].z}, 
                 {pointlist[i][j].x, pointlist[i][j].y, pointlist[i][j].z},
                 {pointlist[i][j + 1].x, pointlist[i][j + 1].y, pointlist[i][j + 1].z},
                 {pointlist[i][j + 2].x, pointlist[i][j + 2].y, pointlist[i][j + 2].z});

      if (false && (VPdist(pointlist[i][j], pointlist[i + 1][j]) * 255 > 150 || VPdist(pointlist[i][j], pointlist[i - 1][j]) * 255 > 150)) {
        pclp.r = 255;
        pclp.g = 0;
        pclp.b = 0;
      }
      else if (false && (f({pointlist[i][j - 1].x, pointlist[i][j - 1].y, pointlist[i][j - 1].z}, 
                 {pointlist[i][j].x, pointlist[i][j].y, pointlist[i][j].z},
                 {pointlist[i][j + 1].x, pointlist[i][j + 1].y, pointlist[i][j + 1].z},
                 {pointlist[i][j + 2].x, pointlist[i][j + 2].y, pointlist[i][j + 2].z}) * 255 > 150)) {
        pclp.r = 0;
        pclp.g = 0;
        pclp.b = 255;
      }
      else {
        pclp.r = std::min(f_value, 20.0f) / 20.0f * 255;
        pclp.g = std::min(f_value, 20.0f) / 20.0f * 255;
        pclp.b = std::min(f_value, 20.0f) / 20.0f * 255;
      }
      cloud.push_back(pclp);
    }
  }
}

void pcap_hdl32_segment(std::string pcap) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::visualization::CloudViewer viewer("Stream");

  std::vector<std::vector<VPoint>> pointlist;
  VelodyneStreamer vs;
  vs.open(pcap);

  while (!viewer.wasStopped()) {
    while (vs.nextFrameInOrder(pointlist)) {
      std::cout << pointlist.size() * 32 << std::endl;
      pclcloud->clear();
      pclcloud->resize(0);

      graphCanny(pointlist, *pclcloud);

      viewer.showCloud(pclcloud);
    }
    break;
  }
}

void pcap_hdl64_segment(std::string pcap) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::visualization::CloudViewer viewer("Stream");

  std::vector<std::vector<VPoint>> pointlist;
  VelodyneStreamer vs;
  vs.open(pcap);

  while (!viewer.wasStopped()) {
    while (vs.nextFrameInOrder64(pointlist)) {
      std::cout << pointlist.size() * 64 << std::endl;
      pclcloud->clear();
      pclcloud->resize(0);

      graphCanny(pointlist, *pclcloud);

      viewer.showCloud(pclcloud);
    }
    break;
  }
}


void ASCI_to_BINARY(std::string file_name) {
  std::cout << file_name << std::endl;
  pcl::PointCloud<PCLMyPointType> cloud;
  pcl::PCDReader reader;
  reader.read(file_name, cloud);
  pcl::io::savePCDFileBinary(file_name + "_BIN.pcd", cloud);  
}

int main() {

  VelodyneStreamer vs1;
  vs1.open("/media/bence/Data/DATA/PCAP/VLP16/Autós mérések/20160825/2016-08-25-16-53-18_Velodyne-VLP-16-DataAbsoluteGeo.pcap");
  vs1.sensor = SensorType::VLP16;
  
  VelodyneStreamer vs2;
  vs2.open("/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160825/02.pcap");
  vs2.sensor = SensorType::HDL64;

  VCloud cloud1, cloud2;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud1(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pclcloud2(new pcl::PointCloud<pcl::PointXYZRGB>());

  while(vs1.nextFrame(cloud1) && cloud1.at(0)->timestamp < 3260000000); std::cout << "VLP kiporgetve" << std::endl;
  while(vs2.nextFrame(cloud2) && cloud2.at(0)->timestamp < 3260000000); std::cout << "HDL kiporgetve" << std::endl;

  VCloud CL1, CL2;
  VPoint p;
  while(vs1.nextFrame(cloud1) && cloud1.at(0)->timestamp < 3300000000) {
    for (size_t i = 0; i < cloud1.size(); i++) {
      p.x = cloud1.at(i)->x;
      p.y = cloud1.at(i)->y;
      p.z = cloud1.at(i)->z;
      p.distance = cloud1.at(i)->distance;
      p.timestamp = cloud1.at(i)->timestamp;
      CL1.push_back(p);
    }
  }
  while(vs2.nextFrame(cloud2) && cloud2.at(0)->timestamp < 3300000000) {
    for (size_t i = 0; i < cloud2.size(); i++) {
      p.x = cloud2.at(i)->x;
      p.y = cloud2.at(i)->y;
      p.z = cloud2.at(i)->z;
      p.distance = cloud2.at(i)->distance;
      p.timestamp = cloud2.at(i)->timestamp;
      CL2.push_back(p);
    }
  }
  
  
  pcl::visualization::CloudViewer viewer("c");
  
  unsigned int i1 = 0, i2 = 0;
  unsigned int timestamp_threshold = 3260000000;
  while(true) {

    pclcloud->clear();
    pclcloud->resize(0);

    pclcloud1->clear();
    pclcloud1->resize(0);

    pclcloud2->clear();
    pclcloud2->resize(0);

    pcl::PointXYZRGB p;
           
    float X = 0.0f, Y = 0.0f, Z = 0.0f;
    int number = 0;
    while (CL1.at(i1)->timestamp < timestamp_threshold + 100000 && i1 < CL1.size()) {
      p.x = CL1.at(i1)->x * cosf(-M_PI_2 * 0.925f) - CL1.at(i1)->y * sinf(-M_PI_2 * 0.925f);
      p.y = CL1.at(i1)->x * sinf(-M_PI_2 * 0.925f) + CL1.at(i1)->y * cosf(-M_PI_2 * 0.925f);
      p.z = CL1.at(i1)->z;
      p.r = 255;
      p.g = 0;
      p.b = 0;
      pclcloud1->push_back(p);
      i1++; 

      if (sqrtf(p.x * p.x + p.y * p.y + p.z * p.z) < 1.0f) {
        X = (p.x + number * X) / (number + 1);
        Y = (p.y + number * X) / (number + 1);
        Z = (p.z + number * X) / (number + 1);
        number++;
      }
    }

    while (CL2.at(i2)->timestamp < timestamp_threshold + 100000 && i2 < CL2.size()) {
      p.x = CL2.at(i2)->x + X;
      p.y = CL2.at(i2)->y + Y;
      p.z = CL2.at(i2)->z + Z;
      p.r = 0;
      p.g = 0;
      p.b = 255;
      pclcloud2->push_back(p);
      i2++;
    }
    
 
    for (size_t j = 0; j < pclcloud1->size(); j++) {
      pclcloud->push_back(pclcloud1->at(j));
    }    
    for (size_t j = 0; j < pclcloud2->size(); j++) {
      pclcloud->push_back(pclcloud2->at(j));
    }
    
    viewer.showCloud(pclcloud);
    
    timestamp_threshold += 100000;
    std::this_thread::sleep_for(0.1s);    
  }

  // pcap_hdl64_segment("/media/bence/Data/DATA/PCAP/HDL64/Autós mérések/20160202/2016-02-02-09-37-19_Velodyne-HDL-Data.pcap");
  // pcap_hdl32_segment("/media/bence/Data/DATA/PCAP/HDL32/HDL32-V2_Monterey Highway.pcap");

  return 0;
}
