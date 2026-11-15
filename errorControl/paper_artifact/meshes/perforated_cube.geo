// Coarse boundary-conforming starting mesh for the Hartree treecode.
// Geometry: [-5,5]^3 with a centered cubic cavity [-1,1]^3 removed.
SetFactory("OpenCASCADE");

outer = newv;
Box(outer) = {-5, -5, -5, 10, 10, 10};
cavity = newv;
Box(cavity) = {-1, -1, -1, 2, 2, 2};
cut[] = BooleanDifference{ Volume{outer}; Delete; }{ Volume{cavity}; Delete; };
vol[] = Volume{:};
boundary[] = Boundary{ Volume{vol[]}; };
Physical Volume(1) = {vol[]};
Physical Surface(2) = {boundary[]};

Mesh.CharacteristicLengthMin = 1.5;
Mesh.CharacteristicLengthMax = 2.5;
Mesh.Algorithm3D = 1;
Mesh.ElementOrder = 1;
