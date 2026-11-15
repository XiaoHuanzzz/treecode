// Coarse boundary-conforming starting mesh for the Hartree treecode.
// Geometry: [-5,5]^3 with the upper-right xy quadrant removed through z.
SetFactory("OpenCASCADE");

outer = newv;
Box(outer) = {-5, -5, -5, 10, 10, 10};
notch = newv;
Box(notch) = {0, 0, -5, 5, 5, 10};
cut[] = BooleanDifference{ Volume{outer}; Delete; }{ Volume{notch}; Delete; };
vol[] = Volume{:};
boundary[] = Boundary{ Volume{vol[]}; };
Physical Volume(1) = {vol[]};
Physical Surface(2) = {boundary[]};

Mesh.CharacteristicLengthMin = 1.5;
Mesh.CharacteristicLengthMax = 2.5;
Mesh.Algorithm3D = 1;
Mesh.ElementOrder = 1;
