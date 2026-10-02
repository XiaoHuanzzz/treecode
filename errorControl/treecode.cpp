// Origin treecode. Show CPU times.
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

/// for mesh
#include <AFEPack/HGeometry.h>
/// for template element
#include <AFEPack/TemplateElement.h>
/// for space
#include <AFEPack/DGFEMSpace.h>
/// for operators functionals
#include <AFEPack/Operator.h>
#include <AFEPack/Functional.h>
/// for AMG
#include <AFEPack/AMGSolver.h>


const int DIM = 3;
const double PI = 3.14159265358979323846;
const double algebraic_accuracy = 2;/// !!!!!!!!!!!!!!!!!!!!!!!!! for 3D, only tetrahedron has this number 6. Twin and four tetrahedron, no.

void writeNeighborElementsToFile(const std::string& filename,
                                 const std::set<HElement<DIM>*>& neighbor_of_the_elements,
                                 const RegularMesh<DIM>& r_mesh) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write the number of elements
    file << neighbor_of_the_elements.size() << std::endl;

    // Iterate over the neighbor elements
    for (const auto& hElement : neighbor_of_the_elements) {
        HGeometry<DIM>* current_geometry = hElement->h_element;
        const int n_vtx = current_geometry->vertex.size();

        // Write the number of vertices for this element
        // file << 3 << std::endl;

        // Write the vertices
        for (int j = 0; j <= DIM; ++j) {
            for (int k = 0; k < DIM; ++k) {
                file << r_mesh.point(current_geometry->vertex[j]->index)[k] << " ";
            }
            // file << std::endl;
        }
        file << std::endl;
    }

    file.close();
    // std::cout << "Neighbor elements data written to " << filename << std::endl;
}

double u(const double * p)
{
  return exp(-PI * p[0] * p[0]) * exp(-2. * PI * p[1] * p[1]) * exp(-3. * PI * p[2] * p[2]);
};

double f(const double * p)
{
  return - (4. * PI * PI * p[0] * p[0] + 16. * PI * PI * p[1] * p[1] + 36. * PI * PI * p[2] * p[2] - 12. * PI) * u(p)/ (4. * PI);
};

double tetrahedronVolume(const AFEPack::Point<3>& A, 
  const AFEPack::Point<3>& B, 
  const AFEPack::Point<3>& C, 
  const AFEPack::Point<3>& D) {
// Compute vectors
  AFEPack::Point<3> AB = { B[0] - A[0], B[1] - A[1], B[2] - A[2] };
  AFEPack::Point<3> AC = { C[0] - A[0], C[1] - A[1], C[2] - A[2] };
  AFEPack::Point<3> AD = { D[0] - A[0], D[1] - A[1], D[2] - A[2] };

  // Compute cross product AC × AD
  AFEPack::Point<3> crossProduct = {
    AC[1] * AD[2] - AC[2] * AD[1],
    AC[2] * AD[0] - AC[0] * AD[2],
    AC[0] * AD[1] - AC[1] * AD[0]
  };

  // Compute dot product (AB · (AC × AD))
  double dotProduct = AB[0] * crossProduct[0] + AB[1] * crossProduct[1] + AB[2] * crossProduct[2];

  // Compute volume
  return std::abs(dotProduct) / 6.0;
}

struct hElementCache
{
  std::set<HElement<DIM>* > neighbor_elements;
  std::set<HElement<DIM>* > nonNeighbor_elements;

  int level;
  double area;
  AFEPack::Point<DIM> ref_point;/// by default, we choose barycenter of each hElement
  double D; 
  double monopole;
  std::array<double, DIM> dipole;
  std::array<double, DIM * DIM> quadrupole;
  std::array<double, DIM * DIM * DIM> octupole;
  std::array<double, DIM * DIM * DIM * DIM> hexadecapole;
};

struct ElementCache : public ElementAdditionalData<double, DIM>
{
  HElement<DIM>* the_hElement;/// to build the connection between the element_cache and hElement_cache.
  
  double monopole;
  std::array<double, DIM> dipole;
  std::array<double, DIM * DIM> quadrupole;

};


void construct_neighbor_layers(
    std::set<HGeometry<0, DIM>*>& initial_vertices,
    std::set<HGeometry<0, DIM>*>& outer_vertices_set,
    std::set<HGeometry<0, DIM>*>& vertices_set,
    std::set<HElement<DIM>*>& initial_layer,
    std::set<HElement<DIM>*>& outer_layer,  
    int& initial_level,
    std::set<HElement<DIM>*>& neighbor_of_the_elements,
    std::vector<std::vector<std::set<HElement<DIM>* > > >& node_neighbors,
    int flag            
) { 
    outer_layer.erase(outer_layer.begin(), outer_layer.end());
    outer_vertices_set.erase(outer_vertices_set.begin(), outer_vertices_set.end());
    neighbor_of_the_elements.erase(neighbor_of_the_elements.begin(), neighbor_of_the_elements.end());
    
    if(flag == 1){
      
      // Find neighbors of initial vertices
      for (const auto& vertex : initial_vertices) {
          if (vertex != nullptr) {  // Check if vertex is valid
              int idx_vtx = vertex->index;  // Get index of the vertex
    
              // Ensure idx_vtx is valid before accessing node_neighbors
              if (idx_vtx >= 0 && idx_vtx < node_neighbors.size()) {
                  if (initial_level >= 0 && initial_level < node_neighbors[idx_vtx].size()) {
                      neighbor_of_the_elements.insert(
                          node_neighbors[idx_vtx][initial_level].begin(),
                          node_neighbors[idx_vtx][initial_level].end()
                      );
                  }
              }
              vertices_set.insert(vertex);
          }
      }
  
      //Erase the last layer
      for (auto element : initial_layer) {
          if (element != nullptr) {  // Ensure element is valid
              neighbor_of_the_elements.erase(const_cast<HElement<DIM>*>(element));
          }
      }
  
      std::set<HElement<DIM>*> elements_to_add;  // Temporary container for new elements
    
      // Store siblings in elements_to_add, store parents in outer_layers
      for (auto element : neighbor_of_the_elements) {
          if (element != nullptr && element->parent != nullptr) {
              // outer_layer.insert(element->parent);  // Insert the parent directly
              for (auto child : element->parent->child) {
                  if (child != nullptr) {  // Ensure child is valid
                      elements_to_add.insert(child);  // Add children to temporary set
                  }
              }
          }
      }
  
      for (auto element : initial_layer) {
          if (element != nullptr) {  // Ensure element is valid
              elements_to_add.erase(const_cast<HElement<DIM>*>(element));
          }
      }
  
      // Now merge the elements from `elements_to_add` into `neighbor_of_the_elements`
      neighbor_of_the_elements.insert(elements_to_add.begin(), elements_to_add.end());
    
      
  
      for (auto element : neighbor_of_the_elements) {
          if (element != nullptr && element->parent != nullptr) {
              outer_layer.insert(element->parent); 
          }
      }
  
      // Store outer_layer's vertices in outer_vertices_set
      for (const auto& element : outer_layer) {
          if (element != nullptr) {  // Ensure element is valid
              std::vector<HGeometry<0, DIM>*> vertices = element->h_element->vertex;
              if (vertices.size() > 0) {
                  outer_vertices_set.insert(vertices.begin(), vertices.end());
              }
          }
      }
  
      // Erase initial_vertices in outer_vertices_set
      for (const auto& vertex : vertices_set) {
          if (vertex != nullptr) {  // Ensure vertex is valid
              outer_vertices_set.erase(const_cast<HGeometry<0, DIM>*>(vertex));  // Erase one by one
          }
      }
  }
  else if (flag == 0){
    // Find neighbors of initial vertices
    // initial_level--;
      for (const auto& vertex : initial_vertices) {
          if (vertex != nullptr) {  // Check if vertex is valid
              int idx_vtx = vertex->index;  // Get index of the vertex
    
              // Ensure idx_vtx is valid before accessing node_neighbors
              if (idx_vtx >= 0 && idx_vtx < node_neighbors.size()) {
                  if (initial_level >= 0 && initial_level < node_neighbors[idx_vtx].size()) {
                      neighbor_of_the_elements.insert(
                          node_neighbors[idx_vtx][initial_level].begin(),
                          node_neighbors[idx_vtx][initial_level].end()
                      );
                  }
              }
              vertices_set.insert(vertex);
          }
      }

      //Erase the last layer
      for (auto element : initial_layer) {
        const int& n_child = element->child.size();
      for(int i = 0;i < n_child;++ i){
          if (element->child[i] != nullptr) {  // Ensure element is valid
              neighbor_of_the_elements.erase(const_cast<HElement<DIM>*>(element->child[i]));
          }
      }
    }

      std::set<HElement<DIM>*> elements_to_add;  // Temporary container for new elements
    
      // Store siblings in elements_to_add, store parents in outer_layers
      for (auto element : neighbor_of_the_elements) {
          if (element != nullptr && element->parent != nullptr) {
              // outer_layer.insert(element->parent);  // Insert the parent directly
              for (auto child : element->parent->child) {
                  if (child != nullptr) {  // Ensure child is valid
                      elements_to_add.insert(child);  // Add children to temporary set
                  }

              }
          }
    }

    for (auto element : initial_layer) {
        const int& n_child = element->child.size();
      for(int i = 0;i < n_child;++ i){
          if (element->child[i] != nullptr) {  // Ensure element is valid
            elements_to_add.erase(const_cast<HElement<DIM>*>(element->child[i]));
          }
      }
    }

    // Now merge the elements from `elements_to_add` into `neighbor_of_the_elements`
    neighbor_of_the_elements.insert(elements_to_add.begin(), elements_to_add.end());
  
    

    for (auto element : neighbor_of_the_elements) {
        if (element != nullptr && element->parent != nullptr) {
            outer_layer.insert(element->parent); 
        }
    }

    // Store outer_layer's vertices in outer_vertices_set
    for (const auto& element : outer_layer) {
        if (element != nullptr) {  // Ensure element is valid
            std::vector<HGeometry<0, DIM>*> vertices = element->h_element->vertex;
            if (vertices.size() > 0) {
                outer_vertices_set.insert(vertices.begin(), vertices.end());
            }
        }
    }

    // Erase initial_vertices in outer_vertices_set
    for (const auto& vertex : vertices_set) {
        if (vertex != nullptr) {  // Ensure vertex is valid
            outer_vertices_set.erase(const_cast<HGeometry<0, DIM>*>(vertex));  // Erase one by one
        }
    }
  }
}


void record_current_hElement_node_neighbors(std::vector<hElementCache>& hElement_cache,
    HElement<DIM>& the_HElement,
              int level,
              std::vector<std::vector<std::set<HElement<DIM>* > > >& node_neighbors,
              int& n_hElement)
{
  n_hElement ++;
  HGeometry<DIM>* the_geometry = the_HElement.h_element;
  std::vector<HGeometry<0, DIM>* > the_vertex = the_geometry->vertex;

  const int& n_vtx = the_vertex.size();
  for(int i = 0;i < n_vtx;++ i){
    node_neighbors[the_vertex[i]->index][level].insert(&the_HElement);
  }

  if(the_HElement.value == 0) return;
  else{
    // leaf_elements.push_back(&the_HElement);
    const int& n_child = the_HElement.child.size();
    for(int i = 0;i < n_child;++ i){
      record_current_hElement_node_neighbors(hElement_cache,*the_HElement.child[i], level + 1, node_neighbors, n_hElement);
    }
  }
  
}

void find_all_leaf_elements(HElement<DIM>* current,
                            std::vector<HElement<DIM>*>& leaf_elements)
{
    // Base case: if current is a leaf node (has no children)
    if (current->value == 0) {
        leaf_elements.push_back(current);
        return;
    }

    // Recursive case: iterate through the children
    const std::vector<HElement<DIM>*>& children = current->child;
    for (HElement<DIM>* child : children) {
        find_all_leaf_elements(child, leaf_elements);
    }
}

void find_elements_MAC(RegularMesh<DIM>& r_mesh, 
  HElement<DIM>& the_hElement,
  hElementCache& the_hElement_cache,
  std::vector<hElementCache>& hElement_cache,
  std::map<AFEPack::HElement<DIM>*, int> map_HElement_to_int,
  int& level,
  AFEPack::Point<DIM> ref_point, int& flag)
{
  std::map<HElement<DIM>*, int>::iterator the_map_it = map_HElement_to_int.find(&the_hElement);
  const int& idx_hElementCache = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[idx_hElementCache];
  HGeometry<DIM>* the_hGeometry = the_hElement.h_element;
  std::vector<HGeometry<0, DIM> *> the_vtx = the_hGeometry->vertex;

  AFEPack::Point<DIM> A = r_mesh.point(the_vtx[0]->index);
  AFEPack::Point<DIM> B = r_mesh.point(the_vtx[1]->index);
  AFEPack::Point<DIM> C = r_mesh.point(the_vtx[2]->index);
  AFEPack::Point<DIM> D = r_mesh.point(the_vtx[3]->index);

  AFEPack::Point<DIM> current_point = A+B+C+D;
  current_point /= 4.0;

  AFEPack::Point<DIM> diff = current_point - ref_point;
  double distance = diff.length();

  AFEPack::Point<DIM> diffA = current_point - A;
  double distanceA = diffA.length();
  AFEPack::Point<DIM> diffB = current_point - B;
  double distanceB = diffB.length();
  AFEPack::Point<DIM> diffC = current_point - C;
  double distanceC = diffC.length();
  AFEPack::Point<DIM> diffD = current_point - D;
  double distanceD = diffD.length();

  double max_distance = std::max({distanceA, distanceB, distanceC, distanceD});
  if (max_distance/distance > 1.0/1.0){
    // down to the next level
    flag = 0;
    const int& n_child = the_hElement.child.size(); // cautious!
    if(the_hElement.value == 0){
      the_hElement_cache.neighbor_elements.insert(&the_hElement);
      return;
    }
    else{
      #pragma omp parallel for
      for(int i = 0;i < n_child;++ i){
          find_elements_MAC(r_mesh, *(the_hElement.child[i]), the_hElement_cache, hElement_cache, map_HElement_to_int, level, ref_point, flag);
      }
    }
  }
  else{
    // add it to nonNeighbor
    the_hElement_cache.nonNeighbor_elements.insert(&the_hElement);
    return;
  }
}

void fill_neighbor_elements_for_current_hElement(RegularMesh<DIM>& r_mesh, 
  std::vector<HElement<DIM> *> root_element_list,
  HElement<DIM>& the_hElement,
    int& the_idx,
    std::vector<hElementCache>& hElement_cache,
    std::map<HElement<DIM>*, int>& map_HElement_to_int,
    int& level,
    std::vector<std::vector<std::set<HElement<DIM>* > > >& node_neighbors)
    {
        //std::cout << "The current the_idx is " << the_idx << std::endl;
        map_HElement_to_int.insert({&the_hElement, the_idx});/// create a pair in map
        hElementCache& the_hElement_cache = hElement_cache[the_idx];
        HGeometry<DIM>* the_hGeometry = the_hElement.h_element;
        std::vector<HGeometry<0, DIM> *> the_vtx = the_hGeometry->vertex;
    
        AFEPack::Point<DIM> A = r_mesh.point(the_vtx[0]->index);
        AFEPack::Point<DIM> B = r_mesh.point(the_vtx[1]->index);
        AFEPack::Point<DIM> C = r_mesh.point(the_vtx[2]->index);
        AFEPack::Point<DIM> D = r_mesh.point(the_vtx[3]->index);
        AFEPack::Point<DIM> ref_point = A+B+C+D;
        ref_point /= 4.0;
        AFEPack::Point<DIM> dA = A-ref_point;
        AFEPack::Point<DIM> dB = B-ref_point;
        AFEPack::Point<DIM> dC = C-ref_point;
        AFEPack::Point<DIM> dD = D-ref_point;
        double da = dA.length();
        double db = dB.length();
        double dc = dC.length();
        double dd = dD.length();
        double length = std::max({da, db, dc, dd});
        the_hElement_cache.D = length;
        double area = tetrahedronVolume(A, B, C, D);
        the_hElement_cache.area = area;
  
          std::set<HGeometry<0, DIM>*> initial_vertices;
          std::vector<HGeometry<0, DIM>*> vertices = the_hElement.h_element->vertex;
          initial_vertices.insert(vertices.begin(), vertices.end());
      
      // Store siblings in elements_to_add, store parents in outer_layers
      std::set<HGeometry<0, DIM>*> outer_vertices_set;
        std::set<HGeometry<0, DIM>*> vertices_set;
          std::set<HElement<DIM>*> neighbor_of_the_elements;
          std::set<HElement<DIM>*> nonneighbor_of_the_elements;
          std::set<HElement<DIM>*> outer_layer;
          std::set<HElement<DIM>*> initial_layer;
          initial_layer.insert(&the_hElement);
      
          construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, initial_layer, outer_layer, level, neighbor_of_the_elements, node_neighbors, 1);
          for(auto element : neighbor_of_the_elements){
            std::vector<HElement<DIM>*> leaf_elements;
            find_all_leaf_elements(element, leaf_elements);
            for(auto ele : leaf_elements)
              hElement_cache[the_idx].neighbor_elements.insert(ele);
          }
          hElement_cache[the_idx].nonNeighbor_elements.insert(root_element_list.begin(), root_element_list.end());
      
          hElement_cache[the_idx].neighbor_elements.insert(&the_hElement);
          int _level = level;
          
            initial_vertices.erase(initial_vertices.begin(), initial_vertices.end());
            for(auto vertex : outer_vertices_set){
              initial_vertices.insert(vertex);
            }
            initial_layer.erase(initial_layer.begin(), initial_layer.end());
            for(auto element : outer_layer){
              initial_layer.insert(element);
            }
            construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, initial_layer, outer_layer, _level, neighbor_of_the_elements, node_neighbors, 1);
            for(auto element : neighbor_of_the_elements){
              std::vector<HElement<DIM>*> leaf_elements;
              find_all_leaf_elements(element, leaf_elements);
              for(auto ele : leaf_elements)
                hElement_cache[the_idx].neighbor_elements.insert(ele);
            }
            // _level--;
              initial_vertices.erase(initial_vertices.begin(), initial_vertices.end());
            for(auto vertex : outer_vertices_set){
              initial_vertices.insert(vertex);
            }
            initial_layer.erase(initial_layer.begin(), initial_layer.end());
            for(auto element : outer_layer){
              initial_layer.insert(element);
            }
            construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, initial_layer, outer_layer, _level, neighbor_of_the_elements, node_neighbors, 1);
            for(auto element : neighbor_of_the_elements){
              std::vector<HElement<DIM>*> leaf_elements;
              find_all_leaf_elements(element, leaf_elements);
              for(auto ele : leaf_elements)
                hElement_cache[the_idx].neighbor_elements.insert(ele);
            }
            
          int flag = 1;
          while(_level > 0){
            _level--;
              initial_vertices.erase(initial_vertices.begin(), initial_vertices.end());
            for(auto vertex : outer_vertices_set){
              initial_vertices.insert(vertex);
            }
            initial_layer.erase(initial_layer.begin(), initial_layer.end());
            for(auto element : outer_layer){
              initial_layer.insert(element);
            }
            construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, initial_layer, outer_layer, _level, neighbor_of_the_elements, node_neighbors, flag);
            for(auto element : neighbor_of_the_elements){
              flag = 1;
              // find_elements_MAC(r_mesh, *element, the_hElement_cache, hElement_cache, map_HElement_to_int, _level, ref_point, flag);
              hElement_cache[the_idx].nonNeighbor_elements.insert(element);
            }
             if(flag == 0)
                _level++; 
              
            }
        
        for(auto element : hElement_cache[the_idx].nonNeighbor_elements){
            HElement<DIM>* element_parent = element;
            if(element_parent->parent == nullptr){
                continue;
            }
            while(element_parent->parent != nullptr){
              element_parent = element_parent->parent;
            }
            hElement_cache[the_idx].nonNeighbor_elements.erase(const_cast<HElement<DIM>*>(element_parent));
          }
    
        for(auto element : hElement_cache[the_idx].neighbor_elements){
          HElement<DIM>* element_parent = element;
          if(element_parent->parent == nullptr){
              continue;
          }
          while(element_parent->parent != nullptr){
            element_parent = element_parent->parent;
          }
          hElement_cache[the_idx].nonNeighbor_elements.erase(const_cast<HElement<DIM>*>(element_parent));
        }

        writeNeighborElementsToFile("neighbor_elements.txt", hElement_cache[the_idx].neighbor_elements, r_mesh);
    
        if(the_hElement.value == 0) {/// if this is a leaf node
          return;
        }
        else{
            const int& n_child = the_hElement.child.size();
            for(int i = 0;i < n_child;++ i){
                fill_neighbor_elements_for_current_hElement(r_mesh, root_element_list, *(the_hElement.child[i]),
                    ++ the_idx,
                    hElement_cache,
                    map_HElement_to_int,
                    ++ level,
                    node_neighbors);
                -- level;      
                }
        }
    }
    

/// Attention: Besides generating nonNeighbor_elements, we also revise neighbor_elements.
void fill_nonNeighbor_elements_for_current_hElement(HElement<DIM>& the_hElement,
    std::vector<HElement<DIM> *> root_element_list,
    HElement<DIM>& current_root,
                std::vector<hElementCache>& hElement_cache,
                std::set<HElement<DIM>* >& complete_set,
                std::map<HElement<DIM>*, int>& map_HElement_to_int,
                std::vector<std::vector<std::set<HElement<DIM>* > > >& node_neighbors,
              int level)
{
  
 
  int _level = level;
  
  std::map<HElement<DIM>*, int>::iterator the_map_it = map_HElement_to_int.find(&(the_hElement));
  const int& idx_hElement = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[idx_hElement];

  hElement_cache[idx_hElement].level = level;
//    the_hElementCache.nonNeighbor_elements.insert(root_element_list[0]);
//    the_hElementCache.nonNeighbor_elements.erase(&current_root);
  // the_hElementCache.nonNeighbor_elements.insert(root_element_list.begin(), root_element_list.end());
        //  std::cout << "nonneighbor = " << hElement_cache.back().nonNeighbor_elements.size() << std::endl;
          
  //the_hElementCache.nonNeighbor_elements.erase(const_cast<HElement<DIM>*>(&current_root));
        //  std::cout << "nonneighbor = " << hElement_cache.back().nonNeighbor_elements.size() << std::endl;
        

  // store vertices of the helement
  std::set<HGeometry<0, DIM>*> initial_vertices;
        std::vector<HGeometry<0, DIM>*> vertices = the_hElement.h_element->vertex;
        initial_vertices.insert(vertices.begin(), vertices.end());

//   // store vertices of the helement


std::set<HGeometry<0, DIM>*> outer_vertices_set;
std::set<HGeometry<0, DIM>*> vertices_set;
std::set<HElement<DIM>*> neighbor_of_the_elements;
std::set<HElement<DIM>*> nonneighbor_of_the_elements;
std::set<HElement<DIM>*> outer_layer;
std::set<HElement<DIM>*> initial_layer;

// Insert the HElement into the initial layer
 initial_layer.insert(&the_hElement);

// First construction of neighbor layers
//  construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, initial_layer, outer_layer, level, neighbor_of_the_elements, node_neighbors, 0);

// std::cout << "neighbor = " << hElement_cache[idx_hElement].neighbor_elements.size() << std::endl;
// while(_level > 0){
//     _level--;
//     initial_vertices.erase(initial_vertices.begin(), initial_vertices.end());
//     for(auto vertex : outer_vertices_set){
//       initial_vertices.insert(vertex);
//     }
//     initial_layer.erase(initial_layer.begin(), initial_layer.end());
//     for(auto element : outer_layer){
//       initial_layer.insert(element);
//     }
    
//     // Construct the neighbor layers for the current level
//     construct_neighbor_layers(initial_vertices, outer_vertices_set, vertices_set, initial_layer, outer_layer, _level, nonneighbor_of_the_elements, node_neighbors, 0);

//     // Insert non-neighbor elements into hElement_cache
//     for(auto element : nonneighbor_of_the_elements){
//         hElement_cache[idx_hElement].nonNeighbor_elements.insert(element);
//     }
    
// }
// // Now process the parent-child relationships, and erase parent elements from nonNeighbor_elements    
// //Track the parent-child relationships and prepare elements for deletion
// for(auto element : the_hElementCache.nonNeighbor_elements){
//     HElement<DIM>* element_parent = element;
//     if(element_parent->parent == nullptr){
//         continue;
//     }
//     while(element_parent->parent != nullptr){
//       element_parent = element_parent->parent;
//     }
//     the_hElementCache.nonNeighbor_elements.erase(const_cast<HElement<DIM>*>(element_parent));
//   }

// for(auto element : the_hElementCache.neighbor_elements){
//     HElement<DIM>* element_parent = element;
    
//     // Traverse to the root parent
//     while(element_parent->parent != nullptr){
//         element_parent = element_parent->parent;
//     }
    
//     // Mark the parent element to be erased
//     the_hElementCache.nonNeighbor_elements.erase(const_cast<HElement<DIM>*>(element_parent));
// }
// std::cout << "nonneighbor" <<  hElement_cache[idx_hElement].nonNeighbor_elements.size() << std::endl;


// double area = 0.0;
// for (auto element : the_hElementCache.neighbor_elements){
//   std::map<HElement<DIM>*, int>::iterator tmp_map_it = map_HElement_to_int.find(element);
// const int& tmpCache_idx = tmp_map_it->second;
// hElementCache& tmp_hElementCache = hElement_cache[tmpCache_idx];
// area += tmp_hElementCache.area;
// // std:: cout << tmp_hElementCache.area << std::endl;
// }
// for (auto element : the_hElementCache.nonNeighbor_elements){
//   std::map<HElement<DIM>*, int>::iterator tmp_map_it = map_HElement_to_int.find(element);
// const int& tmpCache_idx = tmp_map_it->second;
// hElementCache& tmp_hElementCache = hElement_cache[tmpCache_idx];
// area += tmp_hElementCache.area;
// // std:: cout << tmp_hElementCache.area << std::endl;

// }
// std:: cout << area << std::endl;

    //  hElement_cache[idx_hElement].neighbor_elements.insert(&the_hElement);


    if(the_hElement.value == 0){///if this is a leaf node
      return;
    }
  
        const int& n_child = the_hElement.child.size();
        
  /// now we are ready for a recursive call
  for(int i = 0;i < n_child;++ i){
    fill_nonNeighbor_elements_for_current_hElement(*(the_hElement.child[i]),
    root_element_list,
    current_root,
              hElement_cache,
              complete_set,
              map_HElement_to_int,
              node_neighbors,
            ++level);
  }
  
}

void write_current_hElement_neighbor(std::fstream& file,
            std::fstream& mpFile,
            HElement<DIM>* the_hElement,
            std::map<HElement<DIM>*, int>& map_HElement_to_int,
            std::vector<hElementCache>& hElement_cache,
            RegularMesh<DIM>& r_mesh
            )
{
  std::map<HElement<DIM>* , int>::iterator the_map_it = map_HElement_to_int.find(the_hElement);
  const int& idx_hElementCache = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[idx_hElementCache];

  //std::set<HElement<DIM>* >& the_neighbor_elements = the_hElementCache.neighbor_elements;
  std::set<HElement<DIM>* >::iterator
    the_set_it = the_hElementCache.neighbor_elements.begin(),
    end_set_it = the_hElementCache.neighbor_elements.end();      
  for(;the_set_it != end_set_it;++ the_set_it){
    HGeometry<DIM>* current_geometry = (*the_set_it)->h_element;
    const int& n_vtx = current_geometry->vertex.size();
    mpFile << "draw ";
    for(int j = 0;j < n_vtx;++ j){
      if(j != n_vtx - 1){
  mpFile << "(" << 10. *r_mesh.point(current_geometry->vertex[j]->index)[0] << "cm, " 
        << 10.* r_mesh.point(current_geometry->vertex[j]->index)[1] << "cm)--";
      }
      else{
  mpFile << "(" << 10.* r_mesh.point(current_geometry->vertex[j]->index)[0] << "cm, " 
        << 10.* r_mesh.point(current_geometry->vertex[j]->index)[1] << "cm)--cycle;" << std::endl;
      }
      
      for(int k = 0;k < DIM;++ k){
  file << r_mesh.point(current_geometry->vertex[j]->index)[k] << " ";
      }
      file << std::endl;
    }
  }

  if(the_hElement->parent == NULL) return;

  write_current_hElement_neighbor(file, mpFile, the_hElement->parent, map_HElement_to_int, hElement_cache, r_mesh);
}

void write_current_layer(std::fstream & file,
      int& level,
      HElement<DIM>* the_hElement,
      std::map<HElement<DIM>*, int>& map_HElement_to_int,
      std::vector<hElementCache>& hElement_cache,
      RegularMesh<DIM>& r_mesh
      )
{
  if(the_hElement->parent != NULL){
    write_current_layer(file,
      ++ level,
      the_hElement->parent,
      map_HElement_to_int,
      hElement_cache,
      r_mesh);
  }

  int current_level_for_plot = 1;

  if(level < current_level_for_plot) return;
  std::map<HElement<DIM>*, int>::iterator the_map_it = map_HElement_to_int.find(the_hElement);
  const int& idx_hElementCache = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[idx_hElementCache];
  std::set<HElement<DIM>* >& neighbor_elements = the_hElementCache.neighbor_elements;
  std::set<HElement<DIM>* >& nonNeighbor_elements = the_hElementCache.nonNeighbor_elements;

  //file << "beginfig(" << level << ")" << std::endl;

  std::set<HElement<DIM>* >::iterator the_set_it, end_set_it;
  the_set_it = neighbor_elements.begin();
  end_set_it = neighbor_elements.end();    
  for(;the_set_it != end_set_it;++ the_set_it){
    HElement<DIM>* current_hElement = *the_set_it;
    HGeometry<DIM>* current_hGeometry = current_hElement->h_element;
    std::vector<HGeometry<0, DIM>* > the_vtx = current_hGeometry->vertex;
    const int& n_vtx = the_vtx.size();

    file << "fill ";
    for(int j = 0;j < n_vtx;++ j){
      if(j != n_vtx - 1){
  file << std::fixed <<std::setprecision(16) << "("
      << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
      << r_mesh.point(the_vtx[j]->index)[1] << "cm)--";
      }
      else{
  file << "("
      << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
      << r_mesh.point(the_vtx[j]->index)[1] << "cm)--cycle withcolor white;"
      << std::endl;
      }      
    }
    file << "draw ";
    for(int j = 0;j < n_vtx;++ j){
      if(j != n_vtx - 1){
  file << "("
      << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
      << r_mesh.point(the_vtx[j]->index)[1] << "cm)--";
      }
      else{
  file << "("
      << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
      << r_mesh.point(the_vtx[j]->index)[1] << "cm)--cycle;"
      << std::endl;
      }      
    }

  }

  
  the_set_it = nonNeighbor_elements.begin();
  end_set_it = nonNeighbor_elements.end();
  for(;the_set_it != end_set_it;++ the_set_it){
    HElement<DIM>* current_hElement = *the_set_it;
    HGeometry<DIM>* current_hGeometry = current_hElement->h_element;
    std::vector<HGeometry<0, DIM>* > the_vtx = current_hGeometry->vertex;
    const int& n_vtx = the_vtx.size();

    file << "fill ";
    for(int j = 0;j < n_vtx;++ j){
      if(j != n_vtx - 1){
  file << "("
      << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
      << r_mesh.point(the_vtx[j]->index)[1] << "cm)--";
      }
      else{
  if(level == current_level_for_plot){
    file << "("
        << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
        << r_mesh.point(the_vtx[j]->index)[1] << "cm)--cycle withcolor 0." << 7 << "white;"
        << std::endl;
  }
  else{
    file << "("
        << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
        << r_mesh.point(the_vtx[j]->index)[1] << "cm)--cycle withcolor white;"
        << std::endl;
  }
      }      
    }
    file << "draw ";
    for(int j = 0;j < n_vtx;++ j){
      if(j != n_vtx - 1){
  file << "("
      << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
      << r_mesh.point(the_vtx[j]->index)[1] << "cm)--";
      }
      else{
  if(level == current_level_for_plot){	
    file << "("
        << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
        << r_mesh.point(the_vtx[j]->index)[1] << "cm)--cycle;"
        << std::endl;
  }
  else{
    file << "("
        << r_mesh.point(the_vtx[j]->index)[0] << "cm, "
        << r_mesh.point(the_vtx[j]->index)[1] << "cm)--cycle dashed evenly scaled 8;"
        << std::endl;
  }
      }      
    }

  }

  level --;

  //file << "endfig;" << std::endl;

  // if(the_hElement->parent == NULL) return;

  // write_current_layer(file,
  // 		      ++ level,
  // 		      the_hElement->parent,
  // 		      map_HElement_to_int,
  // 		      hElement_cache,
  // 		      r_mesh);

  
}

void update_hElementCache(IrregularMesh<DIM>& ir_mesh,
        std::vector<hElementCache>& hElement_cache,
        std::map<HElement<DIM>*, int>& map_HElement_to_int
             
            )
{
  RegularMesh<DIM>& r_mesh = ir_mesh.regularMesh();

  

  const int n_node = r_mesh.n_geometry(0);

  std::cout << "There are total " << n_node << " nodes" << std::endl;

  /// we will build a neighbor element list for every node in the
  /// mesh. In this list, we also split the information into different
  /// layers, i.e., root layer, root + 1 layer, root + 2 layer, etc,
  /// since we need to find the neighbor elements in each layers.
  /// 1. By default, we introduce 10 layers. More layers will be added when needed.
  /// 2. This list will be updated when the mesh is updated.
  std::vector<std::vector<std::set<HElement<DIM>* > > > node_neighbors(n_node,
                      std::vector<std::set<HElement<DIM>*> >(10));

  std::vector<HElement<DIM> *> root_element_list;
  IrregularMesh<DIM>::RootIterator
    the_root = ir_mesh.beginRootElement(),
    end_root = ir_mesh.endRootElement();
  for(;the_root != end_root;++ the_root){
    root_element_list.push_back(&(*the_root));
  }
  const int& n_root_element = root_element_list.size();


  int n_hElement = 0;
  the_root = ir_mesh.beginRootElement();
  for(;the_root != end_root;++ the_root){
    int level = 0;
    record_current_hElement_node_neighbors(hElement_cache,*the_root, level, node_neighbors, n_hElement);
  }

  

  std::cout << "There are total " << n_hElement << " hElements" << std::endl;
  hElement_cache.resize(n_hElement);


  double b_time = omp_get_wtime();  
  /// fill neighbor_elements in each HElement first.
  int the_idx = 0;
  int level = 0;  
  //the_root = ir_mesh.beginRootElement();
  //for(;the_root != end_root;++ the_root){
  /// can not parallelize following code, due to the_idx, the index for increasing hElement_cache
  for(int i = 0;i < n_root_element;++ i){
    HElement<DIM>* current_root = root_element_list[i];    
    fill_neighbor_elements_for_current_hElement(r_mesh, root_element_list, *current_root, the_idx, hElement_cache, map_HElement_to_int, level, node_neighbors);
    ++ the_idx;
  }

  std::cout << "Filling neighbor_elements uses: " << omp_get_wtime() - b_time << " seconds" << std::endl;

  b_time = omp_get_wtime();  
  /// fill nonneighrbor_elements in each HElement, then.
  std::set<HElement<DIM> *> complete_set;
  
  
  std::cout << "Filling root nonNeighbor_elements uses: " << omp_get_wtime() - b_time << " seconds" << std::endl;


int _level = 0;
  b_time = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp for private(complete_set)
    for(int i = 0;i < n_root_element;++ i){
      HElement<DIM>* current_root = root_element_list[i];
      HElement<DIM>* root = root_element_list[i];
      fill_nonNeighbor_elements_for_current_hElement(*root,
        root_element_list,
        *current_root,
                hElement_cache,
                complete_set,
                map_HElement_to_int,
                node_neighbors,
              _level);
    }
  }

  
  std::cout << "fill all nonRoot nonNeighbor_elements uses: " << omp_get_wtime() - b_time << " seconds " << std::endl;
  //  getchar();


  int tmp = 0;   
  std::fstream mFile;
  level = 1;
  mFile.open("treeCode.mp", std::ios::in|std::ios::out|std::ios::trunc);/// metapost, tikz
  mFile << "beginfig(1)" << std::endl;
  IrregularMesh<DIM>::ActiveIterator
    the_activeElement = ir_mesh.beginActiveElement(),
    end_activeElement = ir_mesh.endActiveElement();      
  for(;the_activeElement != end_activeElement;++ the_activeElement){
    tmp ++;
    if(tmp < 2300) continue;
    write_current_layer(mFile,
      level,
      &(*the_activeElement),
      map_HElement_to_int,
      hElement_cache,
      r_mesh
      );

    HGeometry<DIM>* the_hGeometry = (*the_activeElement).h_element;
    std::vector<HGeometry<0, DIM>* >& the_vtx = the_hGeometry->vertex;
    const int& n_vtx = the_vtx.size();
    mFile << "fill ";
    for(int i = 0;i < n_vtx;++ i){
      if(i != n_vtx - 1){
  mFile << "(" << r_mesh.point(the_vtx[i]->index)[0]
        << "cm, " << r_mesh.point(the_vtx[i]->index)[1] << "cm)--";
      }
      else{
  mFile << "(" << r_mesh.point(the_vtx[i]->index)[0]
        << "cm, " << r_mesh.point(the_vtx[i]->index)[1] << "cm)--cycle withcolor 0.9red;"
        <<std::endl;
      }
    }
    break;
  }

  mFile << "endfig;" << std::endl;
  mFile << "bye;" << std::endl;
  mFile.close();

}


void updateElementCache(std::vector<ElementCache > & element_cache,
      FEMSpace<double, DIM>& fem_space)
{
  element_cache.clear();
  
  const int& n_element = fem_space.n_element();
  element_cache.resize(n_element);

  const Mesh<DIM>&  r_mesh = fem_space.mesh();

  #pragma omp parallel for
  for(int i = 0;i < n_element;++ i){
    Element<double, DIM>& the_ele = fem_space.element(i);
    const int& ele_idx = the_ele.index();

    ElementCache& the_ec = element_cache[ele_idx];
    GeometryBM& the_geo = the_ele.geometry();

    const int& n_vtx = the_geo.n_vertex();
    std::vector<AFEPack::Point<DIM> > vertex(n_vtx);
    for(int j = 0;j < n_vtx;++ j){
      vertex[j] = r_mesh.point()[the_geo.vertex()[j]]; //
    }
    
    int& n_quadrature_point = the_ec.n_quadrature_point;
    double& volume = the_ec.volume;
    AFEPack::Point<DIM>& bc = the_ec.bc;
    std::vector<double>& Jxw = the_ec.Jxw;
    std::vector<AFEPack::Point<DIM> >& q_point = the_ec.q_point;

    std::vector<std::vector<double> >& basis_value = the_ec.basis_value;
    std::vector<std::vector<std::vector<double> > >& basis_gradient = the_ec.basis_gradient;


    barycenter(vertex, bc);
    
    double Tvolume = the_ele.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = the_ele.findQuadratureInfo(algebraic_accuracy); //
    std::vector<double> jacobian = the_ele.local_to_global_jacobian(quad_info.quadraturePoint());
    n_quadrature_point = quad_info.n_quadraturePoint();
    q_point = the_ele.local_to_global(quad_info.quadraturePoint());

    basis_value = the_ele.basis_function_value(q_point);
    basis_gradient = the_ele.basis_function_gradient(q_point);

    volume = 0.;
    Jxw.resize(n_quadrature_point);
    for (int l = 0;l < n_quadrature_point;l ++) {
      Jxw[l] = quad_info.weight(l)*jacobian[l]*Tvolume;
      volume += Jxw[l];      
    }
    
  }
}

// --- helper: compute signed determinant of 3x3 matrix [a b c] (columns)
double det3(const AFEPack::Point<DIM>& a,
            const AFEPack::Point<DIM>& b,
            const AFEPack::Point<DIM>& c) {
    // treat points as vectors (x,y,z). Using column form
    double A00 = a[0], A10 = a[1], A20 = a[2];
    double A01 = b[0], A11 = b[1], A21 = b[2];
    double A02 = c[0], A12 = c[1], A22 = c[2];
    return A00*(A11*A22 - A12*A21)
         - A01*(A10*A22 - A12*A20)
         + A02*(A10*A21 - A11*A20);
}

void calculate_multipole(double& monopole,
            std::array<double, DIM>& dipole,
            std::array<double, DIM * DIM>& quadrupole,
            std::array<double, DIM * DIM * DIM>& octupole,
            std::array<double, DIM * DIM * DIM * DIM>& hexadecapole,
            std::vector<double>& Jxw,
            AFEPack::Point<DIM>& ref_point, 
            const std::vector<AFEPack::Point<DIM>> q_point,
          int n_quadrature_point){
  for (int l = 0; l < n_quadrature_point; ++l) {
      double f_val = f(q_point[l]);
      double dx = q_point[l][0] - ref_point[0];
      double dy = q_point[l][1] - ref_point[1];
      double dz = q_point[l][2] - ref_point[2];
      double rp_length_2 = dx*dx + dy*dy + dz*dz;

      monopole += Jxw[l] * f_val;
      dipole[0] += Jxw[l] * f_val * dx;
      dipole[1] += Jxw[l] * f_val * dy;
      dipole[2] += Jxw[l] * f_val * dz;

      quadrupole[0] += Jxw[l] * f_val * (3.0 * dx * dx - rp_length_2);
      quadrupole[1] += Jxw[l] * f_val * (3.0 * dx * dy);
      quadrupole[2] += Jxw[l] * f_val * (3.0 * dx * dz);
      quadrupole[3] += Jxw[l] * f_val * (3.0 * dy * dx);
      quadrupole[4] += Jxw[l] * f_val * (3.0 * dy * dy - rp_length_2);
      quadrupole[5] += Jxw[l] * f_val * (3.0 * dy * dz);
      quadrupole[6] += Jxw[l] * f_val * (3.0 * dz * dx);
      quadrupole[7] += Jxw[l] * f_val * (3.0 * dz * dy);
      quadrupole[8] += Jxw[l] * f_val * (3.0 * dz * dz - rp_length_2);

      for (int i = 0; i < 3; ++i) {
        double dxi = (i == 0) ? dx : (i == 1) ? dy : dz;
        for (int j = 0; j < 3; ++j) {
          double dxj = (j == 0) ? dx : (j == 1) ? dy : dz;
          for (int k = 0; k < 3; ++k) {
            double dxk = (k == 0) ? dx : (k == 1) ? dy : dz;
            int idx3 = i + 3 * j + 9 * k;
            octupole[idx3] += Jxw[l] * f_val * dxi * dxj * dxk;

            for (int m = 0; m < 3; ++m) {
              double dxm = (m == 0) ? dx : (m == 1) ? dy : dz;
              int idx4 = i + 3 * j + 9 * k + 27 * m;
              hexadecapole[idx4] += Jxw[l] * f_val * dxi * dxj * dxk * dxm;
            }
          }
        }
      }
    }
    return;
}


// Split one tetrahedron (v0, v1, v2, v3) into 8 smaller tetrahedra
// Output: vector of 8 tetrahedra, each tetrahedron is std::array<AFEPack::Point<3>, 4>
std::vector<std::array<AFEPack::Point<3>, 4>> split_tetrahedron_8(
    const AFEPack::Point<3>& v0,
    const AFEPack::Point<3>& v1,
    const AFEPack::Point<3>& v2,
    const AFEPack::Point<3>& v3)
{
    // 1️⃣ compute midpoints of all 6 edges
    AFEPack::Point<3> m01 = (v0 + v1);
    m01 *= 0.5;
    AFEPack::Point<3> m02 = (v0 + v2);
    m02 *= 0.5;
    AFEPack::Point<3> m03 = (v0 + v3);
    m03 *= 0.5;
    AFEPack::Point<3> m12 = (v1 + v2);
    m12 *= 0.5;
    AFEPack::Point<3> m13 = (v1 + v3);
    m13 *= 0.5;
    AFEPack::Point<3> m23 = (v2 + v3);
    m23 *= 0.5;

    // 2️⃣ define 8 subtetrahedra (each has 4 vertices)
    std::vector<std::array<AFEPack::Point<3>, 4>> subtets(8);

    subtets[0] = {v0, m01, m02, m03};
    subtets[1] = {v1, m01, m12, m13};
    subtets[2] = {v2, m02, m12, m23};
    subtets[3] = {v3, m03, m13, m23};

    subtets[4] = {m01, m02, m03, m12};
    subtets[5] = {m01, m03, m12, m13};
    subtets[6] = {m02, m03, m12, m23};
    subtets[7] = {m12, m13, m23, m03};

    return subtets;
}

// Build quadrature (4-point degree-2) and call calculate_multipole
void multipole_from_vertices(
    double& monopole,
    std::array<double, DIM>& dipole,
    std::array<double, DIM * DIM>& quadrupole,
    std::array<double, DIM * DIM * DIM>& octupole,
    std::array<double, DIM * DIM * DIM * DIM>& hexadecapole,
    const AFEPack::Point<DIM>& v0,
    const AFEPack::Point<DIM>& v1,
    const AFEPack::Point<DIM>& v2,
    const AFEPack::Point<DIM>& v3)
{
    // 1) compute volume
    AFEPack::Point<DIM> e1, e2, e3;
    for (int i = 0; i < DIM; ++i) {
        e1[i] = v1[i] - v0[i];
        e2[i] = v2[i] - v0[i];
        e3[i] = v3[i] - v0[i];
    }
    double detB = det3(e1, e2, e3);            // det of mapping matrix
    double volume = std::abs(detB) / 6.0;      // tetra volume

    // 2) choose quadrature: 4-point symmetric degree-2
    const double a = 0.1381966011250105; // parameter
    const double lam[4][4] = {
      {a, a, a, 1.0 - 3.0*a},
      {a, a, 1.0 - 3.0*a, a},
      {a, 1.0 - 3.0*a, a, a},
      {1.0 - 3.0*a, a, a, a}
    };
    const int n_qp = 4;

    // 3) build q_point (physical) and Jxw
    std::vector<AFEPack::Point<DIM>> q_point;
    std::vector<double> Jxw;
    q_point.reserve(n_qp);
    Jxw.reserve(n_qp);

    for (int qi = 0; qi < n_qp; ++qi) {
        AFEPack::Point<DIM> x;
        // barycentric to physical: x = sum lambda_i * v_i
        for (int d = 0; d < DIM; ++d) {
            x[d] = lam[qi][0] * v0[d]
                 + lam[qi][1] * v1[d]
                 + lam[qi][2] * v2[d]
                 + lam[qi][3] * v3[d];
        }
        q_point.push_back(x);
        Jxw.push_back(volume / static_cast<double>(n_qp)); // equally distributed
    }

    // 4) pick ref_point (centroid recommended)
    AFEPack::Point<DIM> ref_point;
    for (int d = 0; d < DIM; ++d) {
        ref_point[d] = (v0[d] + v1[d] + v2[d] + v3[d]) / 4.0;
    }

    // 5) initialize multipoles to zero (if not already)
    monopole = 0.0;
    for (int i = 0; i < DIM; ++i) dipole[i] = 0.0;
    for (auto &x: quadrupole) x = 0.0;
    for (auto &x: octupole) x = 0.0;
    for (auto &x: hexadecapole) x = 0.0;

    // 6) call calculate_multipole
    calculate_multipole(monopole, dipole, quadrupole, octupole, hexadecapole,
                        Jxw, ref_point, q_point, n_qp);
}




void collect_multipole_from_leaf_hElement(double& monopole,
            std::array<double, DIM>& dipole,
            std::array<double, DIM * DIM>& quadrupole,
            std::array<double, DIM * DIM * DIM>& octupole,
            std::array<double, DIM * DIM * DIM * DIM>& hexadecapole,
            AFEPack::Point<DIM>& ref_point,
            std::vector<ElementCache>& element_cache,
            HElement<DIM>* the_hElement)
{
  if (the_hElement->value == 0) {
    const int& ele_idx = the_hElement->index;
    ElementCache& the_ec = element_cache[ele_idx];
    std::vector<AFEPack::Point<DIM>>& q_point = the_ec.q_point;
    std::vector<double>& Jxw = the_ec.Jxw;
    const int& n_quadrature_point = the_ec.n_quadrature_point;

    // calculate_multipole(monopole, dipole, quadrupole, octupole, hexadecapole, Jxw, ref_point, q_point, n_quadrature_point);

    for (int l = 0; l < n_quadrature_point; ++l) {
      double f_val = f(q_point[l]);
      double dx = q_point[l][0] - ref_point[0];
      double dy = q_point[l][1] - ref_point[1];
      double dz = q_point[l][2] - ref_point[2];
      double rp_length_2 = dx*dx + dy*dy + dz*dz;

      monopole += Jxw[l] * f_val;
      dipole[0] += Jxw[l] * f_val * dx;
      dipole[1] += Jxw[l] * f_val * dy;
      dipole[2] += Jxw[l] * f_val * dz;

      quadrupole[0] += Jxw[l] * f_val * (3.0 * dx * dx - rp_length_2);
      quadrupole[1] += Jxw[l] * f_val * (3.0 * dx * dy);
      quadrupole[2] += Jxw[l] * f_val * (3.0 * dx * dz);
      quadrupole[3] += Jxw[l] * f_val * (3.0 * dy * dx);
      quadrupole[4] += Jxw[l] * f_val * (3.0 * dy * dy - rp_length_2);
      quadrupole[5] += Jxw[l] * f_val * (3.0 * dy * dz);
      quadrupole[6] += Jxw[l] * f_val * (3.0 * dz * dx);
      quadrupole[7] += Jxw[l] * f_val * (3.0 * dz * dy);
      quadrupole[8] += Jxw[l] * f_val * (3.0 * dz * dz - rp_length_2);

      for (int i = 0; i < 3; ++i) {
        double dxi = (i == 0) ? dx : (i == 1) ? dy : dz;
        for (int j = 0; j < 3; ++j) {
          double dxj = (j == 0) ? dx : (j == 1) ? dy : dz;
          for (int k = 0; k < 3; ++k) {
            double dxk = (k == 0) ? dx : (k == 1) ? dy : dz;
            int idx3 = i + 3 * j + 9 * k;
            octupole[idx3] += Jxw[l] * f_val * dxi * dxj * dxk;

            for (int m = 0; m < 3; ++m) {
              double dxm = (m == 0) ? dx : (m == 1) ? dy : dz;
              int idx4 = i + 3 * j + 9 * k + 27 * m;
              hexadecapole[idx4] += Jxw[l] * f_val * dxi * dxj * dxk * dxm;
            }
          }
        }
      }
    }
    return;
  }
  else {
    const int& n_child = the_hElement->child.size();
    for (int i = 0; i < n_child; ++i) {
      collect_multipole_from_leaf_hElement(monopole, dipole, quadrupole, octupole, hexadecapole, ref_point, element_cache, the_hElement->child[i]);
    }
  }
}


void update_multipoleInfo_hElementCache(HElement<DIM>* the_hElement,
          std::vector<hElementCache>& hElement_cache,
          std::map<HElement<DIM>*, int>& map_HElement_to_int,
          IrregularMesh<DIM>& ir_mesh,
          std::vector<ElementCache>& element_cache)
{
  /// 1. prepare the hElementCache
  std::map<HElement<DIM>*, int>::iterator the_map_it = map_HElement_to_int.find(the_hElement);
  const int& idx_hElementCache = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[idx_hElementCache];

  RegularMesh<DIM>& r_mesh = ir_mesh.regularMesh();

  /// 2. update hElementCache.ref_point
  AFEPack::Point<DIM>& ref_point = the_hElementCache.ref_point;
  for(int j = 0;j < DIM;++ j){
    ref_point[j] = 0.;
  }
  HGeometry<DIM>* the_hGeometry = the_hElement->h_element;
  const int& n_vtx = the_hGeometry->vertex.size();
  for(int i = 0;i < n_vtx;++ i){
    AFEPack::Point<DIM>& the_vtx = r_mesh.point(the_hGeometry->vertex[i]->index);//// ONLY for uniform mesh
    for(int j = 0;j < DIM;++ j){
      ref_point[j] += the_vtx[j];
    }
  }
  for(int j = 0;j < DIM;++ j){
    ref_point[j] /= n_vtx;
  }

  /// 3. tracing back to all leaf hElement, and collect the useful
  /// information from those leaf hElement.

  double& monopole = the_hElementCache.monopole;
  std::array<double, DIM>& dipole = the_hElementCache.dipole;
  std::array<double, DIM * DIM>& quadrupole = the_hElementCache.quadrupole;
  std::array<double, DIM * DIM * DIM>& octupole = the_hElementCache.octupole;
  std::array<double, DIM * DIM * DIM * DIM>& hexadecapole = the_hElementCache.hexadecapole;


  monopole = 0.;/// initialize
  dipole[0] = 0.; dipole[1] = 0.; dipole[2] = 0.;
  quadrupole[0] = 0.; quadrupole[1] = 0.; quadrupole[2] = 0.;
  quadrupole[3] = 0.; quadrupole[4] = 0.; quadrupole[5] = 0.;
  quadrupole[6] = 0.; quadrupole[7] = 0.; quadrupole[8] = 0.;
  octupole[0] = 0.; octupole[1] = 0.; octupole[2] = 0.;
  octupole[3] = 0.; octupole[4] = 0.; octupole[5] = 0.;
  octupole[6] = 0.; octupole[7] = 0.; octupole[8] = 0.;
  octupole[10] = 0.; octupole[11] = 0.; octupole[12] = 0.;
  octupole[13] = 0.; octupole[14] = 0.; octupole[15] = 0.;
  octupole[16] = 0.; octupole[17] = 0.; octupole[18] = 0.;
  octupole[20] = 0.; octupole[21] = 0.; octupole[22] = 0.;
  octupole[23] = 0.; octupole[24] = 0.; octupole[25] = 0.;
  octupole[26] = 0.; octupole[9] = 0.; octupole[19] = 0.;
  for(int i = 0; i < DIM*DIM*DIM*DIM; i++){
    hexadecapole[i] = 0.;
  }

  collect_multipole_from_leaf_hElement(monopole, dipole, quadrupole, octupole, hexadecapole, ref_point, element_cache, the_hElement);
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////

  if(the_hElement->value == 0){/// if this is a leaf node
    return;
  }
  else{
    const int& n_child = the_hElement->child.size();
    for(int i = 0;i < n_child;++ i){
        update_multipoleInfo_hElementCache(the_hElement->child[i],
          hElement_cache,
          map_HElement_to_int,
          ir_mesh,
          element_cache);
      
    }
  }
}

int countMAC = 0;

void calculate_nonNeighbor_hElement_multipole_contribution(double & approximation,
                const double* target_point,
                HElement<DIM>* the_hElement,
                std::map<HElement<DIM>*, int>& map_HElement_to_int,
                std::vector<hElementCache>& hElement_cache,
              RegularMesh<DIM>& r_mesh)
{
  std::map<HElement<DIM>*, int>::iterator the_map_it = map_HElement_to_int.find(the_hElement);
  const int& hElementCache_idx = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[hElementCache_idx];

  std::set<HElement<DIM>* >& the_nonNeighbor_elements = the_hElementCache.nonNeighbor_elements; //
  std::set<HElement<DIM>* >& the_Neighbor_elements = the_hElementCache.neighbor_elements; //
  // Combine neighbor and non-neighbor elements into one set
std::set<HElement<DIM>*> all_elements;

// Insert all elements from non-neighbor set
all_elements.insert(the_hElementCache.nonNeighbor_elements.begin(),
                    the_hElementCache.nonNeighbor_elements.end());

// Insert all elements from neighbor set
all_elements.insert(the_hElementCache.neighbor_elements.begin(),
                    the_hElementCache.neighbor_elements.end());

  std::set<HElement<DIM>* >::iterator the_set_it, end_set_it;  
  the_set_it = the_nonNeighbor_elements.begin();
  end_set_it = the_nonNeighbor_elements.end();
  for(;the_set_it != end_set_it;++ the_set_it){
    the_map_it = map_HElement_to_int.find(*the_set_it);
    const int& nonNeighbor_hElementCache_idx = the_map_it->second;
    hElementCache& the_nonNeighbor_hElementCache = hElement_cache[nonNeighbor_hElementCache_idx];
    const AFEPack::Point<DIM>& ref_point = the_nonNeighbor_hElementCache.ref_point;
    double R = 0;
    for(int i = 0; i < DIM; i++){
      R += (ref_point[i]-target_point[i])*(ref_point[i]-target_point[i]);
    }
    R = std::sqrt(R);
    const double r = the_nonNeighbor_hElementCache.D;
    const double& monopole = the_nonNeighbor_hElementCache.monopole;
      const std::array<double, DIM>& dipole = the_nonNeighbor_hElementCache.dipole;
      const std::array<double, DIM * DIM>& quadrupole = the_nonNeighbor_hElementCache.quadrupole;
      const std::array<double, DIM * DIM * DIM>& octupole = the_nonNeighbor_hElementCache.octupole;
      const std::array<double, DIM * DIM * DIM * DIM>& hexadecapole = the_nonNeighbor_hElementCache.hexadecapole;
    
    // if(r / R >= 1){
    //   std::cout << r << " " << R << std::endl;
    //   std::cout << "MAC warning " << std::endl;
    //   countMAC ++;

    //   // writeNeighborElementsToFile("all_elements.txt", all_elements, r_mesh);
    //   // std::ofstream file("all_elements.txt", std::ios::app);
    //   //  file << target_point[0] << ' ' << target_point[1] << ' ' << target_point[2] <<  " mark" << std::endl;
    //   //   file << ref_point[0] << ' ' << ref_point[1] << ' ' << ref_point[2] << " mark" << std::endl;
    //   // std::exit(0);
    //   double monopole = 0.0;
    //   std::array<double, DIM> dipole = {0.0};
    //   std::array<double, DIM * DIM> quadrupole = {0.0};
    //   std::array<double, DIM * DIM * DIM> octupole = {0.0};
    //   std::array<double, DIM * DIM * DIM * DIM> hexadecapole = {0.0};
    //   HElement<DIM>* helem = *the_set_it;

    //   std::vector<HGeometry<0, DIM>*> initial_vertices = helem->h_element->vertex;
    //   std::vector<std::array<AFEPack::Point<3>, 4>> split_vertices = split_tetrahedron_8(*initial_vertices[0], *initial_vertices[1], *initial_vertices[2], *initial_vertices[3]);
    //   for(int i = 0; i < 8; i++){
    //   multipole_from_vertices(monopole, dipole, quadrupole, octupole, hexadecapole,
    //                     split_vertices[i][0], split_vertices[i][1],
    //                      split_vertices[i][2], split_vertices[i][3]);
    //   }
      
    // } 
    
    double xp_length_2 =
      (target_point[0] - ref_point[0]) * (target_point[0] - ref_point[0]) +
      (target_point[1] - ref_point[1]) * (target_point[1] - ref_point[1]) +
      (target_point[2] - ref_point[2]) * (target_point[2] - ref_point[2]);
    double xp_length = pow(xp_length_2, 0.5);

    double tmp_app = 0.0;
    double dx = target_point[0] - ref_point[0];
    double dy = target_point[1] - ref_point[1];
    double dz = target_point[2] - ref_point[2];

    double octupole_contraction = 0.0;
    for (int i = 0; i < 3; ++i) {
        double ri = (i == 0) ? dx : (i == 1) ? dy : dz;
        for (int j = 0; j < 3; ++j) {
            double rj = (j == 0) ? dx : (j == 1) ? dy : dz;
            for (int k = 0; k < 3; ++k) {
                double rk = (k == 0) ? dx : (k == 1) ? dy : dz;
                int idx = i + 3 * j + 9 * k;
                octupole_contraction += octupole[idx] * ri * rj * rk;
            }
        }
    }

    double hexadecapole_contraction = 0.0;
    for (int i = 0; i < 3; ++i) {
        double ri = (i == 0) ? dx : (i == 1) ? dy : dz;
        for (int j = 0; j < 3; ++j) {
            double rj = (j == 0) ? dx : (j == 1) ? dy : dz;
            for (int k = 0; k < 3; ++k) {
                double rk = (k == 0) ? dx : (k == 1) ? dy : dz;
                for (int m = 0; m < 3; ++m) {
                    double rm = (m == 0) ? dx : (m == 1) ? dy : dz;
                    int idx4 = i + 3 * j + 9 * k + 27 * m;
                    hexadecapole_contraction += hexadecapole[idx4] * ri * rj * rk * rm;
                }
            }
        }
    }

    tmp_app = 
      (
      1./xp_length * monopole
      +
      1./pow(xp_length, 3.0) * (
              dx * dipole[0] +
              dy * dipole[1] +
              dz * dipole[2]
              )
      +
      1./(6. * pow(xp_length, 5.0)) *
      (
        (3.* dx * dx - xp_length_2) * quadrupole[0] +
        (3.* dx * dy) * quadrupole[1] +
        (3.* dx * dz) * quadrupole[2] +
        (3.* dy * dx) * quadrupole[3] +       
        (3.* dy * dy - xp_length_2) * quadrupole[4] +
        (3.* dy * dz) * quadrupole[5] +
        (3.* dz * dx) * quadrupole[6] +
        (3.* dz * dy) * quadrupole[7] +       
        (3.* dz * dz - xp_length_2) * quadrupole[8]
        )
      +
      1. / (6. * pow(xp_length, 7.0)) * octupole_contraction
      +
      1. / (24. * pow(xp_length, 9.0)) * hexadecapole_contraction
      );

    approximation += tmp_app;  
  }

  return;
}


double multipole_approximation(const double * target_point,
            HElement<DIM>* the_hElement,
            std::map<HElement<DIM>*, int>& map_HElement_to_int,
            std::vector<hElementCache>& hElement_cache,
            std::vector<ElementCache>& element_cache,
          RegularMesh<DIM>& r_mesh)
{
  double approximation = 0.;
  std::map<HElement<DIM>*, int>::iterator the_map_it = map_HElement_to_int.find(the_hElement);
  const int& hElementCache_idx = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[hElementCache_idx];
  //AFEPack::Point<DIM>& ref_point = the_hElementCache.ref_point;

  std::set<HElement<DIM>* >& the_neighbor_hElement = the_hElementCache.neighbor_elements;
  //  std::set<HElement<DIM>* > the_nonNeighbor_hElement = the_hElementCache.nonNeighbor_elements;
  std::set<HElement<DIM>* >::iterator the_set_it, end_set_it;

  the_set_it = the_neighbor_hElement.begin();
  end_set_it = the_neighbor_hElement.end();
  for(;the_set_it != end_set_it;++ the_set_it){
    const int& ele_idx = (*the_set_it)->index;
    //if(ele_idx <= element_cache.size()){
      ElementCache& the_ec = element_cache[ele_idx];
      std::vector<double>& Jxw = the_ec.Jxw;
      std::vector<AFEPack::Point<DIM> >& q_point = the_ec.q_point;
      const int& n_quadrature_point = the_ec.n_quadrature_point;

      for(int l = 0;l < n_quadrature_point;++ l){
          double tmp_app = 0.0;
          tmp_app = Jxw[l] *
          f(q_point[l])/pow((q_point[l][0] - target_point[0]) * (q_point[l][0] - target_point[0]) +
                (q_point[l][1] - target_point[1]) * (q_point[l][1] - target_point[1]) +
                (q_point[l][2] - target_point[2]) * (q_point[l][2] - target_point[2]), 0.5);
        approximation += tmp_app;
         }

    }
    calculate_nonNeighbor_hElement_multipole_contribution(approximation,
                target_point,
                the_hElement,
                map_HElement_to_int,
                hElement_cache, r_mesh);

    
    
  return approximation;
}

double direct_integral(const double * target_point,
  HElement<DIM>* the_hElement,
  std::vector<ElementCache>& element_cache_u,
  FEMSpace<double,DIM>& fem_space_u){
    const int& n_element = element_cache_u.size();
    double approximation = 0.0;
#pragma omp parallel for reduction(+:approximation)
  for(int i = 0;i < n_element;++ i){

    Element<double, DIM>& the_ele = fem_space_u.element(i);
    ElementCache& the_ec = element_cache_u[i];
    HElement<DIM>* the_hElement = the_ec.the_hElement;
    
   double Tvolume = the_ele.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = the_ele.findQuadratureInfo(2);
    std::vector<double> jacobian = the_ele.local_to_global_jacobian(quad_info.quadraturePoint());
    const int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<AFEPack::Point<DIM> > q_point = the_ele.local_to_global(quad_info.quadraturePoint());
    
    std::vector<double>& Jxw = the_ec.Jxw;
    for(int l = 0;l < n_quadrature_point;++ l){
      double tmp_app = 0.0;
          tmp_app = Jxw[l] *
          f(q_point[l])/pow((q_point[l][0] - target_point[0]) * (q_point[l][0] - target_point[0]) +
                (q_point[l][1] - target_point[1]) * (q_point[l][1] - target_point[1]) +
                (q_point[l][2] - target_point[2]) * (q_point[l][2] - target_point[2]), 0.5);
        approximation += tmp_app;
    }
  }
  return approximation;
}


bool pointInTetra(const AFEPack::Point<DIM>& p,
                  const AFEPack::Point<DIM>& A,
                  const AFEPack::Point<DIM>& B,
                  const AFEPack::Point<DIM>& C,
                  const AFEPack::Point<DIM>& D)
{
    double V  = (tetrahedronVolume(A,B,C,D));
    double V1 = (tetrahedronVolume(p,B,C,D));
    double V2 = (tetrahedronVolume(A,p,C,D));
    double V3 = (tetrahedronVolume(A,B,p,D));
    double V4 = (tetrahedronVolume(A,B,C,p));
    double diff = fabs(V - (V1+V2+V3+V4));
    // std::cout << "diff = " << diff << std::endl;
    if(diff < 1e-10)
      return 1;
    else
      return 0;
}


HElement<DIM>* findLeafContainingPoint(std::vector<HElement<DIM> *>& root_element_list,
                            AFEPack::Point<DIM>& P,
                            RegularMesh<DIM>& r_mesh,
                            std::map<HElement<DIM>*, int>& map_HElement_to_int)
{
  int n = 0;
  for(int i = 0; i < root_element_list.size(); i++){
    HElement<DIM>* current = root_element_list[i];
    HGeometry<DIM>* geom = current->h_element;
    auto vtx = geom->vertex;

    AFEPack::Point<DIM> A = r_mesh.point(vtx[0]->index);
    AFEPack::Point<DIM> B = r_mesh.point(vtx[1]->index);
    AFEPack::Point<DIM> C = r_mesh.point(vtx[2]->index);
    AFEPack::Point<DIM> D = r_mesh.point(vtx[3]->index);

    if (pointInTetra(P,A,B,C,D)) {
      n = i; 
      
        break;
    }
    if(i == root_element_list.size()-1){
      throw std::runtime_error("Point not inside any root tetrahedron!");
    }
}
// std::cout << n <<std::endl;
    HElement<DIM>* current = root_element_list[n];
    while (!current->child.empty()) {
        bool found = false;
        for (auto* child : current->child) {
            if (child == nullptr) continue;

            HGeometry<DIM>* geom = child->h_element;
            auto vtx = geom->vertex;

            AFEPack::Point<DIM> A = r_mesh.point(vtx[0]->index);
            AFEPack::Point<DIM> B = r_mesh.point(vtx[1]->index);
            AFEPack::Point<DIM> C = r_mesh.point(vtx[2]->index);
            AFEPack::Point<DIM> D = r_mesh.point(vtx[3]->index);

            if (pointInTetra(P,A,B,C,D)) {
                current = child;
                return current;
            }
        }
        if (!found) {
            throw std::runtime_error("Point not inside any child tetrahedron!");
        }
    }

    // Now 'current' is the leaf that contains P
    // int idx = map_HElement_to_int[current];
    
}


double _u_b_(const double *p) {
  double r2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
  double eps = 1e-4;     // make smaller for more singularity
  return exp(-r2/eps);
}

double _f_(const double *p) {
  double r2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
  double eps = 1e-4;
  return (4.0*r2/(eps*eps) - 6.0/eps) * exp(-r2/eps);
}

void get_indicator(Indicator<DIM>& ind,
        RegularMesh<DIM>& mesh, FEMSpace<double,DIM>& fem_space,  std::vector<TemplateElement<double,DIM,DIM> >& template_elements)
  {
 
    StiffMatrix<DIM,double> stiff_matrix(fem_space);
    stiff_matrix.algebricAccuracy() = 6;
    stiff_matrix.build();

    std::cout << "Stiff matrix is built" << std::endl;  

    FEMFunction<double,DIM> solution(fem_space);
    Vector<double> right_hand_side;
    Operator::L2Discretize(&_f_, fem_space, right_hand_side, 2);

    std::cout << "RHS is built" << std::endl;    

    BoundaryFunction<double,DIM> boundary1(BoundaryConditionInfo::DIRICHLET, 1, &_u_b_);
    BoundaryConditionAdmin<double,DIM> boundary_admin(fem_space);
    boundary_admin.add(boundary1);
    boundary_admin.apply(stiff_matrix, solution, right_hand_side);

    std::cout << "Boundary condition is done" << std::endl;      

    AMGSolver solver;
    solver.lazyReinit(stiff_matrix);
    solver.solve(solution, right_hand_side, 1.0e-08, 200);	

    solution.writeOpenDXData("u.dx");
 
    ind.reinit(mesh);
    int n_element = mesh.n_geometry(DIM);
    for(int i = 0;i < n_element;++ i){

      Element<double, DIM>& the_ele = fem_space.element(i);
      GeometryBM& the_geo = the_ele.geometry();
      const int& the_geo_idx = the_geo.index();
    
      double volume = the_ele.templateElement().volume();
      const QuadratureInfo<DIM>& quad_info = the_ele.findQuadratureInfo(2);
      std::vector<double> jacobian = the_ele.local_to_global_jacobian(quad_info.quadraturePoint());
      int n_quadrature_point = quad_info.n_quadraturePoint();
      std::vector<AFEPack::Point<DIM> > q_point = the_ele.local_to_global(quad_info.quadraturePoint());

      for (int l = 0;l < n_quadrature_point;l ++) {
    double Jxw = quad_info.weight(l)*jacobian[l]*volume;
    ind[the_geo_idx] += Jxw * 1./pow(q_point[l][0] * q_point[l][0]
					       + q_point[l][1] * q_point[l][1]
					       + q_point[l][2] * q_point[l][2], 0.5);
      }
    }
  
}


int main(int argc, char* argv[])
{
  HGeometryTree<DIM> h_tree;
  h_tree.readMesh(argv[1]);

  std::unique_ptr<IrregularMesh<DIM>> ir_mesh = std::make_unique<IrregularMesh<DIM>>();

  ir_mesh->reinit(h_tree);
  ir_mesh->semiregularize();
    ir_mesh->regularize(false);
    RegularMesh<DIM>& _r_m = ir_mesh->regularMesh();
    // _r_m.writeOpenDXData("D.dx");
  ir_mesh->globalRefine(5);
  FEMSpace<double,DIM> fem_space;
    std::vector<TemplateElement<double,DIM,DIM> > template_elements;
    int i = 0;
  //do{
    i++;
    /// 对非正则网格做半正则化和正则化
    ir_mesh->semiregularize();
    ir_mesh->regularize(false);

    /// 这就是通过正则化得到的网格
    RegularMesh<DIM>& _r_mesh = ir_mesh->regularMesh();
    // _r_mesh.writeOpenDXData("D.dx");
    
    template_elements.resize(3);
    // std::vector<TemplateElement<double,DIM,DIM> > template_elements(3);
   /// for tetrahedron
  TemplateGeometry<DIM> tet_geo;
  tet_geo.readData("tetrahedron.tmp_geo");
  TemplateDOF<DIM> tet_template_dof(tet_geo);
  tet_template_dof.readData("tetrahedron.1.tmp_dof");
  CoordTransform<DIM> tet_crd_trs;
  tet_crd_trs.readData("tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> tet_basFunctionAdmin(tet_template_dof);
  tet_basFunctionAdmin.readData("tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> tet_unit_outNormal;
  tet_unit_outNormal.readData("tetrahedron.out_nrm");

  template_elements[0].reinit(tet_geo,
			      tet_template_dof,
			      tet_crd_trs,
			      tet_basFunctionAdmin,
			      tet_unit_outNormal);

  /// for twin_tetrahedron
  TemplateGeometry<DIM> twin_tet_geo;
  twin_tet_geo.readData("twin_tetrahedron.tmp_geo");
  TemplateDOF<DIM> twin_tet_template_dof(twin_tet_geo);
  twin_tet_template_dof.readData("twin_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> twin_tet_crd_trs;
  twin_tet_crd_trs.readData("twin_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> twin_tet_basFunctionAdmin(twin_tet_template_dof);
  twin_tet_basFunctionAdmin.readData("twin_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> twin_tet_unit_outNormal;
  twin_tet_unit_outNormal.readData("twin_tetrahedron.out_nrm");

  template_elements[1].reinit(twin_tet_geo,
			      twin_tet_template_dof,
			      twin_tet_crd_trs,
			      twin_tet_basFunctionAdmin,
			      twin_tet_unit_outNormal);

  /// for four_tetrahedron
  TemplateGeometry<DIM> four_tet_geo;
  four_tet_geo.readData("four_tetrahedron.tmp_geo");
  TemplateDOF<DIM> four_tet_template_dof(four_tet_geo);
  four_tet_template_dof.readData("four_tetrahedron.1.tmp_dof");
  CoordTransform<DIM> four_tet_crd_trs;
  four_tet_crd_trs.readData("four_tetrahedron.crd_trs");
  BasisFunctionAdmin<double, DIM> four_tet_basFunctionAdmin(four_tet_template_dof);
  four_tet_basFunctionAdmin.readData("four_tetrahedron.1.bas_fun");
  UnitOutNormal<DIM> four_tet_unit_outNormal;
  four_tet_unit_outNormal.readData("four_tetrahedron.out_nrm");

  template_elements[2].reinit(four_tet_geo,
			      four_tet_template_dof,
			      four_tet_crd_trs,
			      four_tet_basFunctionAdmin,
			      four_tet_unit_outNormal);
 
   //FEMSpace<double,DIM> fem_space(mesh, template_elements);
   
    fem_space.reinit(_r_mesh, template_elements);
   int _n_element = _r_mesh.n_geometry(DIM);
   fem_space.element().resize(_n_element);
   for (int i = 0;i < _n_element;i ++) {
     if (_r_mesh.geometry(DIM,i).n_vertex() == 4) {
       fem_space.element(i).reinit(fem_space,i,0);
     } else if (_r_mesh.geometry(DIM,i).n_vertex() == 5){
       fem_space.element(i).reinit(fem_space,i,1);      
     } else if (_r_mesh.geometry(DIM,i).n_vertex() == 7){
       fem_space.element(i).reinit(fem_space,i,2); 
     } else{} //{std::cout << ""}
   }


   fem_space.buildElement();
   fem_space.buildDof();
   fem_space.buildDofBoundaryMark();

   
//     /// 下面一段计算用于加密的指示子。
//     Indicator<DIM> indicator(_r_mesh);
//     get_indicator(indicator, _r_mesh, fem_space, template_elements);

//     /// 下面的几行调用进行自适应的函数，都是套话。
//     std::unique_ptr<IrregularMesh<DIM>> old_ir_mesh = std::make_unique<IrregularMesh<DIM>>(*ir_mesh);
// ir_mesh = std::make_unique<IrregularMesh<DIM>>(); 

    
//   MeshAdaptor<DIM> mesh_adaptor(*old_ir_mesh, *ir_mesh);
//   mesh_adaptor.setIndicator(indicator);
//   mesh_adaptor.tolerence() = 10.0e-02;
//   mesh_adaptor.convergenceOrder() = 0.;
//   mesh_adaptor.refineStep() = 5;
//   mesh_adaptor.adapt();

//   // } while(i < atoi(argv[2]));   
//   ir_mesh->semiregularize();
//   ir_mesh->regularize(false);
//   RegularMesh<DIM> &r_mesh1 = ir_mesh->regularMesh();
//   r_mesh1.writeOpenDXData("D.dx");
//   fem_space.reinit(r_mesh1, template_elements);
//    fem_space.element().resize(r_mesh1.n_geometry(DIM));
//    for (int i = 0;i < r_mesh1.n_geometry(DIM);i ++) {
//      if (r_mesh1.geometry(DIM,i).n_vertex() == 4) {
//        fem_space.element(i).reinit(fem_space,i,0);
//      } else if (r_mesh1.geometry(DIM,i).n_vertex() == 5){
//        fem_space.element(i).reinit(fem_space,i,1);      
//      } else if (r_mesh1.geometry(DIM,i).n_vertex() == 7){
//        fem_space.element(i).reinit(fem_space,i,2); 
//      } else{} //{std::cout << ""}
//    }


//    fem_space.buildElement();
//    fem_space.buildDof();
//    fem_space.buildDofBoundaryMark();

   
//     // 下面一段计算用于加密的指示子。
//     Indicator<DIM> indicator1(r_mesh1);
//     get_indicator(indicator1, r_mesh1, fem_space, template_elements);

//     /// 下面的几行调用进行自适应的函数，都是套话。
//     std::unique_ptr<IrregularMesh<DIM>> old_ir_mesh1 = std::make_unique<IrregularMesh<DIM>>(*ir_mesh);
// ir_mesh = std::make_unique<IrregularMesh<DIM>>(); 

    
//   MeshAdaptor<DIM> mesh_adaptor1(*old_ir_mesh1, *ir_mesh);
//   mesh_adaptor1.setIndicator(indicator1);
//   mesh_adaptor1.tolerence() = 3.0e-03;
//   mesh_adaptor1.convergenceOrder() = 0.;
//   mesh_adaptor1.refineStep() = 2;
//   mesh_adaptor1.adapt();

//   ir_mesh->semiregularize();
//   ir_mesh->regularize(false);
//   RegularMesh<DIM>& r_mesh2 = ir_mesh->regularMesh();
//   r_mesh2.writeOpenDXData("D.dx");
//     fem_space.reinit(r_mesh2, template_elements);
//    int _n_element2 = r_mesh2.n_geometry(DIM);
//    fem_space.element().resize(_n_element2);
//    for (int i = 0;i < _n_element2;i ++) {
//      if (r_mesh2.geometry(DIM,i).n_vertex() == 4) {
//        fem_space.element(i).reinit(fem_space,i,0);
//      } else if (r_mesh2.geometry(DIM,i).n_vertex() == 5){
//        fem_space.element(i).reinit(fem_space,i,1);      
//      } else if (r_mesh2.geometry(DIM,i).n_vertex() == 7){
//        fem_space.element(i).reinit(fem_space,i,2); 
//      } else{} //{std::cout << ""}
//    }


//    fem_space.buildElement();
//    fem_space.buildDof();
//    fem_space.buildDofBoundaryMark();

//   /// 下面一段计算用于加密的指示子。
//     Indicator<DIM> indicator2(r_mesh2);
//     get_indicator(indicator2, r_mesh2, fem_space, template_elements);

    
  std::vector<hElementCache> hElement_cache;
  std::map<HElement<DIM>*, int> map_HElement_to_int;  
  update_hElementCache(*ir_mesh, hElement_cache, map_HElement_to_int);

  std::vector<ElementCache> element_cache;
  
  updateElementCache(element_cache, fem_space);
  IrregularMesh<DIM>::ActiveIterator
    the_activeElement = ir_mesh->beginActiveElement(),
    end_activeElement = ir_mesh->endActiveElement();      
  for(;the_activeElement != end_activeElement;++ the_activeElement){
    const int& hElement_idx = the_activeElement->index;
    // std::cout << hElement_idx << std::endl;
    ElementCache& the_ec = element_cache[hElement_idx];
    the_ec.the_hElement = &(*the_activeElement);
    
  }
  
  ///////////////////////// update multipole information in each hElement////////////////
  /// what we have: std::vector<hElementCache> hElement_cache;
  RegularMesh<DIM>& r_mesh = ir_mesh->regularMesh();
  std::vector<HElement<DIM> *> root_element_list;
  IrregularMesh<DIM>::RootIterator
    the_root = ir_mesh->beginRootElement(),
    end_root = ir_mesh->endRootElement();
  for(;the_root != end_root;++ the_root){
    root_element_list.push_back(&(*the_root));
  }
  const int& n_root_element = root_element_list.size();

  double b_time = omp_get_wtime();    
#pragma omp parallel for
  for(int i = 0;i < n_root_element;++ i){
    update_multipoleInfo_hElementCache(root_element_list[i], hElement_cache, map_HElement_to_int, *ir_mesh, element_cache);    
  }
  std::cout << "Updating multipoleInfo uses " << omp_get_wtime() - b_time << " seconds " << std::endl;
  double l2_error = 0.;

  //const int& n_element = dg_fem_space.n_element();
  const int& n_element = element_cache.size();
  b_time = omp_get_wtime();      
  auto start = std::chrono::high_resolution_clock::now(); 
#pragma omp parallel for reduction(+:l2_error)
  for(int i = 0;i < n_element;++ i){

    Element<double, DIM>& the_ele = fem_space.element(i);
    ElementCache& the_ec = element_cache[i];
    HElement<DIM>* the_hElement = the_ec.the_hElement;
    if(the_hElement->value!= 0){
      continue;
    }
    // identify the area 
    std::map<HElement<DIM>*, int>::iterator the_map_it = map_HElement_to_int.find(the_hElement);
  const int& hElementCache_idx = the_map_it->second;
  hElementCache& the_hElementCache = hElement_cache[hElementCache_idx];
  
    double Tvolume = the_ele.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = the_ele.findQuadratureInfo(1);
    std::vector<double> jacobian = the_ele.local_to_global_jacobian(quad_info.quadraturePoint());
    const int n_quadrature_point = quad_info.n_quadraturePoint();
    std::vector<AFEPack::Point<DIM> > q_point = the_ele.local_to_global(quad_info.quadraturePoint());

    for(int l = 0;l < n_quadrature_point;++ l){
        double tmp_err = 0.0;
      double approximation = multipole_approximation(q_point[l],
						     the_hElement,
						     map_HElement_to_int,
						     hElement_cache,
						     element_cache, r_mesh);
      // tmp_err = quad_info.weight(l) * jacobian[l] * Tvolume * (u(q_point[l]) - approximation) * (u(q_point[l]) - approximation);   
      // l2_error += tmp_err;
      // if(tmp_err > 0.0001){
      //   std::cout << u(q_point[l]) << " " << approximation << std::endl;
      //   std::cout << q_point[l] << std::endl;
      // }
      // std::cout << "step err = " << tmp_err 
      // // << " volume = " << Tvolume 
      // <<std::endl;   
      // std::cout << i << " " << l2_error << std::endl;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();  // End timer
    std::chrono::duration<double> elapsed = end - start;  // Compute duration

    std::cout << "Execution time: " << elapsed.count() << " seconds\n";
  std::cout << "Doing the calculation uses " << omp_get_wtime() - b_time << " seconds " << std::endl;

  std::cout << "l2_error is " << sqrt(l2_error) << std::endl;

  std::cout << countMAC << std::endl;
  return 0;
}   
