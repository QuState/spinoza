OPENQASM 2.0;
qreg q[4];
h q[0];
cp(-pi) q[0],q[1];
cp(-pi/4) q[0],q[2];
cp(-pi/9) q[0],q[3];
h q[1];
cp(-pi) q[1],q[2];
cp(-pi/4) q[1],q[3];
h q[2];
cp(-pi) q[2],q[3];
h q[3];
