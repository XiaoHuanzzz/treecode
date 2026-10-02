#include <iostream>
#include <AFEPack/HGeometry.h>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <assert.h>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <AFEPack/AMGSolver.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Operator.h>
#include <AFEPack/BilinearOperator.h>
#include <AFEPack/EasyMesh.h>

#include <random>
#include <chrono>

std::vector<int> randomSample(int m, int n) {
    // Create a vector containing numbers from 1 to n
    std::vector<int> population(n);
    for (int i = 0; i < n; ++i) {
        population[i] = i + 1;
    }

    // Seed the random number generator with the current time
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(population.begin(), population.end(), std::default_random_engine(seed));

    // Take the first m elements
    std::vector<int> sample(population.begin(), population.begin() + m);

    return sample;
}


constexpr int DIM = 2; // 二维空间
#define pi 3.1415926535

// 定义密度函数
template<int DIM>
double density(const AFEPack::Point<DIM>& r) {
    // 这里可以定义任意的密度函数
    return exp(-r[0] - r[1]); // 示例：密度函数为 r[0] * r[1]
}

double rho(const double * r) {
    // 这里可以定义任意的密度函数
    return exp(-r[0] - r[1]); // 示例：密度函数为 r[0] * r[1]
}

double _u_b_(const double * p)
{
  return 1.0;
}

/// 右端项
double _f_(const double * p)
{
  return 4*pi*rho(p);
}


// 计算三角形面积
double triangleArea(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C) {
    return 0.5 * std::abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]));
}

// 递归计算monopole
void calculateMonopoleRecursive(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C,
                                double& monopole, double threshold) {
    // 计算当前三角形的面积
    double area = triangleArea(A, B, C);

    // 如果面积小于阈值，停止递归
    if (area < threshold) {
        // 计算当前三角形的monopole
        AFEPack::Point<DIM> centroid = (A + B + C) ;
        centroid /= 3.0;
        double rho = density(centroid);
        monopole += rho * area;
        return;
    }

    // 将三角形剖分成四个小三角形
    AFEPack::Point<DIM> midAB = (A + B) ;
    AFEPack::Point<DIM> midBC = (B + C) ;
    AFEPack::Point<DIM> midCA = (C + A) ;

    midAB /= 2.0;
    midBC /= 2.0;
    midCA /= 2.0;

    // 递归计算四个小三角形的monopole
    calculateMonopoleRecursive(A, midAB, midCA, monopole, threshold);
    calculateMonopoleRecursive(midAB, B, midBC, monopole, threshold);
    calculateMonopoleRecursive(midBC, C, midCA, monopole, threshold);
    calculateMonopoleRecursive(midAB, midBC, midCA, monopole, threshold);
}

// 计算monopole
double calculateMonopole(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C, double threshold) {
    double monopole = 0.0;
    calculateMonopoleRecursive(A, B, C, monopole, threshold);
    return monopole;
}

// 递归计算dipole
void calculateDipoleRecursive(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C,
                              std::array<double, DIM> dipole, const AFEPack::Point<DIM>& barycenter, double threshold) {
    // 计算当前三角形的面积
    double area = triangleArea(A, B, C);

    // 如果面积小于阈值，停止递归
    if (area < threshold) {
        // 计算当前三角形的dipole
        AFEPack::Point<DIM> centroid = (A + B + C) ;
        centroid /= 3.0;
        // std::cout << centroid[0] << ' ' << centroid[1] << std::endl;
        double rho = density(centroid);
        // std::cout << "rho = " << rho << std::endl;
        for (int k = 0; k < DIM; ++k) {
            dipole[k] += (centroid[k]-barycenter[k]) * rho * area;
            // std::cout << "centroid[k]-barycenter[k] = " << centroid[k]-barycenter[k] << std::endl;
        }return;
    }

    // 将三角形剖分成四个小三角形
    AFEPack::Point<DIM> midAB = (A + B) ;
    AFEPack::Point<DIM> midBC = (B + C) ;
    AFEPack::Point<DIM> midCA = (C + A) ;

    midAB /= 2.0;
    midBC /= 2.0;
    midCA /= 2.0;
    // 递归计算四个小三角形的dipole
    calculateDipoleRecursive(A, midAB, midCA, dipole, barycenter, threshold);
    calculateDipoleRecursive(midAB, B, midBC, dipole, barycenter, threshold);
    calculateDipoleRecursive(midBC, C, midCA, dipole, barycenter, threshold);
    calculateDipoleRecursive(midAB, midBC, midCA, dipole, barycenter, threshold);
}

// 计算dipole
std::array<double, DIM> calculateDipole(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C, double threshold) {
    std::array<double, DIM> dipole = {0.0, 0.0};
    AFEPack::Point<DIM> centroid = (A + B + C) ;
        centroid /= 3.0;
    calculateDipoleRecursive(A, B, C, dipole, centroid, threshold);
    return dipole;
}

// 递归计算quadrupole
void calculateQuadrupoleRecursive(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C,
                                  std::array<std::array<double, DIM>, DIM>& quadrupole, const AFEPack::Point<DIM>& barycenter, double threshold) {
    // Calculate the area of the current triangle
    double area = triangleArea(A, B, C);

    // If the area is below the threshold, stop recursion
    if (area < threshold) {
        // Calculate the centroid of the triangle
        AFEPack::Point<DIM> centroid = (A + B + C);
        centroid /= 3.0;

        // Get the density at the centroid
        double rho = density(centroid);

        // Calculate the position vector relative to the barycenter
        AFEPack::Point<DIM> r = centroid - barycenter;

        // Calculate the quadrupole components
        for (int i = 0; i < DIM; ++i) {
            for (int j = 0; j < DIM; ++j) {
                quadrupole[i][j] += rho * area * (3 * r[i] * r[j] - (i == j ? r[0]*r[0]+r[1]*r[1] : 0));
            }
        }
        return;
    }

    // Subdivide the triangle into four smaller triangles
    AFEPack::Point<DIM> midAB = (A + B) ;
    midAB /= 2.0;
    AFEPack::Point<DIM> midBC = (B + C);
    midBC /= 2.0;
    AFEPack::Point<DIM> midCA = (C + A) ;
    midCA /= 2.0;

    // Recursively calculate the quadrupole for the smaller triangles
    calculateQuadrupoleRecursive(A, midAB, midCA, quadrupole, barycenter, threshold);
    calculateQuadrupoleRecursive(midAB, B, midBC, quadrupole, barycenter, threshold);
    calculateQuadrupoleRecursive(midBC, C, midCA, quadrupole, barycenter, threshold);
    calculateQuadrupoleRecursive(midAB, midBC, midCA, quadrupole, barycenter, threshold);
}

// Function to calculate the quadrupole
std::array<std::array<double, DIM>, DIM> calculateQuadrupole(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C, double threshold) {
    std::array<std::array<double, DIM>, DIM> quadrupole = {{{0.0}}};
    AFEPack::Point<DIM> barycenter = (A + B + C) ;
    barycenter /= 3.0;
    calculateQuadrupoleRecursive(A, B, C, quadrupole, barycenter, threshold);
    return quadrupole;
}


// 定义hElementCache结构
struct hElementCache {
    std::set<HElement<DIM>*> neighbor_elements;
    std::set<HElement<DIM>*> nonNeighbor_elements;
    HElement<DIM> root_element;

    AFEPack::Point<DIM> ref_point;
    AFEPack::Point<DIM> barycenter;
    double monopole = 0.0;
    std::array<double, DIM> dipole = {0.0, 0.0};
    std::array<double, DIM * DIM> quadrupole = {0.0, 0.0, 0.0, 0.0};
    int level;

    double area = 0.0;

    double potential = 0.0;
    double integralPotential = 0.0;
};

// 计算并存储monopole, dipole, quadrupole到hElementCache
void calculateAndStoreMoments(const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C, hElementCache& cache, double threshold) {
    cache.monopole = calculateMonopole(A, B, C, threshold);
    cache.dipole = calculateDipole(A, B, C, threshold);
    std::array<std::array<double, DIM>, DIM> quadrupole = calculateQuadrupole(A, B, C, threshold);
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            cache.quadrupole[i * DIM + j] = quadrupole[i][j];
        }
    }
}

// 计算势能
template <int DIM>
double calculatePotential(const AFEPack::Point<DIM>& x, const AFEPack::Point<DIM>& p, const hElementCache& cache) {
    double r_squared = 0.0;
    for (int i = 0; i < DIM; ++i) {
        r_squared += (x[i] - p[i]) * (x[i] - p[i]);
    }
    double r = std::sqrt(r_squared);

    double potential = cache.monopole / r;

    for (int i = 0; i < DIM; ++i) {
        potential += (x[i] - p[i]) * cache.dipole[i] / (r * r * r);
    }

    double quadrupoleTerm = 0.0;
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < DIM; ++j) {
            quadrupoleTerm += (x[i] - p[i]) * cache.quadrupole[i * DIM + j] * (x[j] - p[j]);
        }
    }
    quadrupoleTerm /= (2.0 * r * r * r * r * r);
    potential += quadrupoleTerm;

    return potential;
}

double CalculateIntegral(const AFEPack::Point<DIM>& r, const AFEPack::Point<DIM>& A, const AFEPack::Point<DIM>& B, const AFEPack::Point<DIM>& C, double threshold) {
    double area = triangleArea(A, B, C);

    // Base case: if the area is smaller than the threshold, use midpoint rule
    if (area < threshold) {
        AFEPack::Point<DIM> centroid = (A + B + C) ;
        centroid /= 3.0;
        double distance = (r - centroid).length();
        return density(centroid) * area / distance;
    }

    // Divide the triangle into four smaller triangles
    AFEPack::Point<DIM> midAB = (A + B) ;
    AFEPack::Point<DIM> midBC = (B + C) ;
    AFEPack::Point<DIM> midCA = (C + A) ;

    midAB /= 2.0;
    midBC /= 2.0;
    midCA /= 2.0;
    // Recursively calculate the potential for each sub-triangle
    double potential = 0.0;
    potential += CalculateIntegral(r, A, midAB, midCA, threshold);
    potential += CalculateIntegral(r, midAB, B, midBC, threshold);
    potential += CalculateIntegral(r, midCA, midBC, C, threshold);
    potential += CalculateIntegral(r, midAB, midBC, midCA, threshold);

    if (potential == std::numeric_limits<double>::infinity() || potential ==- std::numeric_limits<double>::infinity()) {
        // std::cout << "Warning: Potential calculation resulted in infinity. Skipping." << std::endl;
        // std::cout << "r = " << r[0] << ' ' << r[1] << std::endl;
        // std::cout << "A = " << A[0] << ' ' << A[1] << std::endl;
        // std::cout << "B = " << B[0] << ' ' << B[1] << std::endl;
        // std::cout << "C = " << C[0] << ' ' << C[1] << std::endl;
        
        return 0.0; // Return 0.0 or another appropriate value to skip this operation
    }



    return potential;
}

void store_integral_in_cache(std::vector<hElementCache>& hElement_cache,
                             const std::map<HElement<DIM>*, int>& map_HElement_to_int,
                             const std::vector<HElement<DIM>*>& leaf_elements,
                             const RegularMesh<DIM>& r_mesh,
                             double threshold, std::vector<int> &sample) { // Removed 'const' here
    std::ofstream outputFile("output.txt", std::ios::out);
    
    #pragma omp parallel for
    for (int j = 0; j < sample.size(); j++) {
        int i = sample[j];
        int cache_index = 0;
        auto it = map_HElement_to_int.find(leaf_elements[i]);
        if (it != map_HElement_to_int.end()) {
            cache_index = it->second;
        }
        hElementCache& cache = hElement_cache[cache_index];
        AFEPack::Point<DIM> r = cache.barycenter; // Reference point for potential calculation

        double totalPotential = 0;

        for (int j = 0; j < leaf_elements.size(); j++) {
            if (i == j)
                continue;
            HGeometry<DIM>* the_geometry = leaf_elements[j]->h_element;
            std::vector<HGeometry<0, DIM>*> the_vertex = the_geometry->vertex;

            AFEPack::Point<DIM> A = r_mesh.point(the_vertex[0]->index);
            AFEPack::Point<DIM> B = r_mesh.point(the_vertex[1]->index);
            AFEPack::Point<DIM> C = r_mesh.point(the_vertex[2]->index);
            int level = cache.level;

            double potential = CalculateIntegral(r, A, B, C, threshold/pow(pow(2,DIM),level));
            // std::cout << "potential = " << potential << std::endl;
            if(potential > 1){
                // std::cout << "warning" << A[0] << ' ' << A[1] << ' ' << B[0] << ' ' << B[1] << ' ' << C[0] << ' ' << C[1] << std::endl;
                continue;
            }
            totalPotential += potential;
        }
        cache.integralPotential = totalPotential;
        // Open a text file for writing
    
    // Check if the file was opened successfully
    if (outputFile.is_open()) {
        // Write the value of the variable to the file
        outputFile << r[0] << ' ' << r[1] << ' ' << totalPotential << std::endl;
        
        // Close the file after writing
        
    }
    
    }
    outputFile.close();
}

void record_current_hElement_node_neighbors(const std::vector<HElement<DIM>*> root_element_list,
                                            HElement<DIM>& root_element,
                                            HElement<DIM>& the_HElement,
                                            int level,
                                            std::vector<std::vector<std::set<HElement<DIM>*>>>& node_neighbors,
                                            int& n_hElement,
                                            std::vector<hElementCache>& hElement_cache,
                                            std::map<HElement<DIM>*, int>& map_HElement_to_int,
                                            std::vector<HElement<DIM>*>& leaf_elements,
                                            const RegularMesh<DIM>& r_mesh,
                                            double threshold) {
    n_hElement++;
    HGeometry<DIM>* the_geometry = the_HElement.h_element;
    std::vector<HGeometry<0, DIM>*> the_vertex = the_geometry->vertex;
    const int& n_vtx = the_vertex.size();
    #pragma omp parallel for
    for (int i = 0; i < n_vtx; ++i) {
        node_neighbors[the_vertex[i]->index][level].insert(&the_HElement);
    }

    // 计算并存储多极矩
    AFEPack::Point<DIM> A = r_mesh.point(the_vertex[0]->index);
    AFEPack::Point<DIM> B = r_mesh.point(the_vertex[1]->index);
    AFEPack::Point<DIM> C = r_mesh.point(the_vertex[2]->index);
    AFEPack::Point<DIM> centroid = (A + B + C) ;
    centroid /= 3.0;
    double area = triangleArea(A, B, C);

    hElementCache cache;
    cache.level = level;
    cache.barycenter = centroid;
    cache.area = area;
    calculateAndStoreMoments(A, B, C, cache, threshold);
    hElement_cache.push_back(cache); // 将cache存储在vector中
    map_HElement_to_int[&the_HElement] = hElement_cache.size() - 1; // 记录HElement到索引的映射

    if (the_HElement.value == 0) {
        leaf_elements.push_back(&the_HElement);
        // Initialize nonNeighbor_elements for leaf elements
        hElement_cache.back().nonNeighbor_elements.insert(root_element_list.begin(), root_element_list.end());
        // std::cout << "nonneighbor = " << hElement_cache.back().nonNeighbor_elements.size() << std::endl;
        
        hElement_cache.back().nonNeighbor_elements.erase(&root_element);
        // std::cout << "nonneighbor = " << hElement_cache.back().nonNeighbor_elements.size() << std::endl;
        
        return;
    } else {
        const int& n_child = the_HElement.child.size();
        for (int i = 0; i < n_child; ++i) {
            record_current_hElement_node_neighbors(root_element_list, root_element, *the_HElement.child[i], level + 1, node_neighbors, n_hElement, hElement_cache, map_HElement_to_int, leaf_elements, r_mesh, threshold);
        }
    }
}

void find_upper_layer(
    HElement<DIM>& leaf_element,
    HElement<DIM>& linear_kin_of_current_layer,
    std::set<HElement<DIM>*>& neighbor_of_the_elements
) {
    // Ensure the reference is properly initialized
    HElement<DIM>* element = nullptr;
    if (linear_kin_of_current_layer.parent != nullptr) {
        element = linear_kin_of_current_layer.parent;
    }

    // If the parent is found, collect its children
    if (element != nullptr) {
        for (auto child : element->child) {
            neighbor_of_the_elements.insert(child);
        }
        // Remove the current layer from the neighbors
        neighbor_of_the_elements.erase(&linear_kin_of_current_layer);
        // Update the current layer to the parent
        linear_kin_of_current_layer = *element;
    }
}

void construct_neighbor_layers(
    HElement<DIM>& initial_element,
    int initial_level,
    std::set<HElement<DIM>*>& neighbor_of_the_elements
) {
    int current_level = initial_level;
    
    HElement<DIM>* linear_kin_of_current_layer = &initial_element;
    
    while (current_level > 0) {
        find_upper_layer(initial_element, *linear_kin_of_current_layer, neighbor_of_the_elements); 
        
        if (linear_kin_of_current_layer->parent == nullptr) {
            // Handle the case where the parent is nullptr (e.g., root element)
            break;
        }
        
        linear_kin_of_current_layer = linear_kin_of_current_layer->parent;
        current_level--;
    }
}

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
        file << n_vtx << std::endl;

        // Write the vertices
        for (int j = 0; j < n_vtx; ++j) {
            for (int k = 0; k < DIM; ++k) {
                file << r_mesh.point(current_geometry->vertex[j]->index)[k] << " ";
            }
            file << std::endl;
        }
    }

    file.close();
    // std::cout << "Neighbor elements data written to " << filename << std::endl;
}

void calculate_treecode_potential_in_cache(
    const std::vector<std::vector<std::set<HElement<DIM>*>>>& node_neighbors,
    std::vector<hElementCache>& hElement_cache,
    const std::map<HElement<DIM>*, int>& map_HElement_to_int,
    const std::vector<HElement<DIM>*>& leaf_elements,
    const RegularMesh<DIM>& r_mesh,
    double threshold, std::vector<int> &sample) {

    // Parallelize the outer loop using OpenMP
    #pragma omp parallel for
    for (int k = 0; k < sample.size(); k++) {
        
        int i = sample[k];
        std::set<HGeometry<0, DIM>*> leaf_vertices_set;
        std::vector<HGeometry<0, DIM>*> vertices = leaf_elements[i]->h_element->vertex;
        leaf_vertices_set.insert(vertices.begin(), vertices.end());

        int cache_index = 0;
        auto it = map_HElement_to_int.find(leaf_elements[i]);
        if (it != map_HElement_to_int.end()) {
            cache_index = it->second;
        }

        int level = hElement_cache[cache_index].level;

        std::set<HElement<DIM>*> neighbor_of_the_elements;
        construct_neighbor_layers(*leaf_elements[i], level, neighbor_of_the_elements);

        // Writing to hElement_cache is safe if cache_index is unique per thread
        hElement_cache[cache_index].neighbor_elements = neighbor_of_the_elements;

        // Ensure thread-safe file writing with a critical section
        #pragma omp critical
        {
            writeNeighborElementsToFile("neighbor_elements.txt", neighbor_of_the_elements, r_mesh);
        }

        double total_potential = 0.0;
        AFEPack::Point<DIM> x = hElement_cache[cache_index].barycenter;

        // Process neighbors (thread-safe if neighbor elements are read-only)
        for (auto neighbor : neighbor_of_the_elements) {
            auto neighbor_it = map_HElement_to_int.find(neighbor);
            if (neighbor_it != map_HElement_to_int.end()) {
                int neighbor_cache_index = neighbor_it->second;
                hElementCache& neighbor_cache = hElement_cache[neighbor_cache_index];
                AFEPack::Point<DIM> p = neighbor_cache.barycenter;

                HGeometry<DIM>* the_geometry = neighbor->h_element;
                std::vector<HGeometry<0, DIM>*> the_vertex = the_geometry->vertex;

                AFEPack::Point<DIM> A = r_mesh.point(the_vertex[0]->index);
                AFEPack::Point<DIM> B = r_mesh.point(the_vertex[1]->index);
                AFEPack::Point<DIM> C = r_mesh.point(the_vertex[2]->index);

                double potential = CalculateIntegral(x, A, B, C, threshold);
                if (x != p && potential < 1) {
                    total_potential += potential;
                }
            }
        }

        // Process non-neighbors
        for (auto element : hElement_cache[cache_index].nonNeighbor_elements) {
            auto nonneighbor_it = map_HElement_to_int.find(element);
            if (nonneighbor_it != map_HElement_to_int.end()) {
                int nonneighbor_cache_index = nonneighbor_it->second;
                hElementCache& nonneighbor_cache = hElement_cache[nonneighbor_cache_index];
                AFEPack::Point<DIM> p_ = nonneighbor_cache.barycenter;
                double potential = calculatePotential(x, p_, hElement_cache[nonneighbor_cache_index]);
                if (potential < 1)
                    total_potential += potential;
            }
        }

        hElement_cache[cache_index].potential = total_potential;

        // Thread-safe output to console
        #pragma omp critical
        {
            std::cout << "total potential = " << total_potential << std::endl;
        }
    }
}

double L2error(std::vector<hElementCache>& hElement_cache,
    const std::map<HElement<DIM>*, int>& map_HElement_to_int,
    const std::vector<HElement<DIM>*>& leaf_elements)
{
    double err = 0.0;
    const int n = leaf_elements.size();
    
    // Parallelize the loop with OpenMP reduction
    #pragma omp parallel for reduction(+:err) schedule(dynamic)
    for(int i = 0; i < n; i++) {
        const auto& element = leaf_elements[i];
        const auto it = map_HElement_to_int.find(element);
        if (it != map_HElement_to_int.end()) {
            const hElementCache& cache = hElement_cache[it->second];
            const double diff = cache.integralPotential - cache.potential;
            err += diff * diff * cache.area;
        }
    }
    
    return std::sqrt(err);
}


int main(int argc, char* argv[]) {
    // 设置递归终止的面积阈值
    double threshold = 1e-6;

    /// 声明几何遗传树，并从Easymesh格式的文件中读入数据
    HGeometryTree<DIM> h_tree;
    h_tree.readEasyMesh("M");

    /// 在背景网格上建立第一个非正则网格，并均匀加密三次
    IrregularMesh<DIM> ir_mesh(h_tree);
    ir_mesh.globalRefine(1);

    /// 对非正则网格做半正则化和正则化
    ir_mesh.semiregularize();
    ir_mesh.regularize(false);


    /// 这就是通过正则化得到的网格
    RegularMesh<DIM>& r_mesh = ir_mesh.regularMesh();

    const int n_node = r_mesh.n_geometry(0);

    std::cout << "There are total " << n_node << " nodes" << std::endl;
    std::vector<hElementCache> hElement_cache; // construct same amount cache
    std::map<HElement<DIM>*, int> map_HElement_to_int; // map helement and the index of cache
    std::vector<HElement<DIM>*> leaf_elements; // store every leaf_elements

    // we will build a neighbor element list for every node in the
    // mesh. In this list, we also split the information into different
    // layers, i.e., root layer, root + 1 layer, root + 2 layer, etc,
    // since we need to find the neighbor elements in each layers.
    // 1. By default, we introduce 10 layers. More layers will be added when needed.
    // 2. This list will be updated when the mesh is updated.
    std::vector<std::vector<std::set<HElement<DIM>*>>> node_neighbors(n_node,
                                                                      std::vector<std::set<HElement<DIM>*>>(10)); //store neighbor

    std::vector<HElement<DIM>*> root_element_list;
    IrregularMesh<DIM>::RootIterator
        the_root = ir_mesh.beginRootElement(),
        end_root = ir_mesh.endRootElement();
    for (; the_root != end_root; ++the_root) {
        root_element_list.push_back(&(*the_root)); // store root element
    }
    const int& n_root_element = root_element_list.size();
    // std::cout << "root element = " << n_root_element << std::endl;
    int n_hElement = 0;
    the_root = ir_mesh.beginRootElement();
    for (; the_root != end_root; ++the_root) {
        int level = 0; // root level 0
        record_current_hElement_node_neighbors(root_element_list, (*the_root), *the_root, level, node_neighbors, n_hElement, hElement_cache,
                                               map_HElement_to_int,
                                               leaf_elements,
                                               r_mesh,
                                               threshold); // from the root recursively record every element is who's neighbor
        
    }

    std::cout << "There are total " << n_hElement << " hElements" << std::endl; // count hElements

    int m = leaf_elements.size()/100;
    int n = leaf_elements.size();
    std::vector<int>sample = randomSample(m, n);

    store_integral_in_cache(hElement_cache,
                            map_HElement_to_int,
                            leaf_elements,
                            r_mesh,
                            threshold, sample);
    for (int i = 0; i < leaf_elements.size(); i++) {
        int cache_index = map_HElement_to_int[leaf_elements[i]];
        hElementCache& cache = hElement_cache[cache_index];
        std::cout << "integral = " << cache.integralPotential << std::endl;
    }
    std::cout << "leaf elements = " << leaf_elements.size() << std::endl;

    calculate_treecode_potential_in_cache(node_neighbors,
                                          hElement_cache,
                                          map_HElement_to_int,
                                          leaf_elements,
                                          r_mesh,
                                          threshold, sample);

    double L2err = L2error(hElement_cache,
                                          map_HElement_to_int,
                                          leaf_elements
                                          );
    std::cout << "L2 error = " << L2err << std::endl;
    
    return 0;
}