// Test-only verbatim original construct_neighbor_layers from
// commit 7038a6c:errorControl/treecode.cpp:125. Do not optimize this oracle.
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
