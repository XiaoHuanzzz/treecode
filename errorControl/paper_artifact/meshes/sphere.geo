// Coarse boundary-conforming starting mesh for the Hartree treecode.
// Geometry: ball of radius 5, centered at the origin.
SetFactory("OpenCASCADE");

Sphere(1) = {0, 0, 0, 5};
vol[] = Volume{:};
boundary[] = Boundary{ Volume{vol[]}; };
Physical Volume(1) = {vol[]};
Physical Surface(2) = {boundary[]};

Mesh.CharacteristicLengthMin = 0.45;
Mesh.CharacteristicLengthMax = 0.75;
Mesh.MeshSizeFromCurvature = 32;
Mesh.Algorithm3D = 1;
Mesh.ElementOrder = 1;
