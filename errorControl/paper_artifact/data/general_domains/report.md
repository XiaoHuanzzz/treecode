# Irregular-domain scaling study with p=2 and P=4

All runs use p=2, P=4, theta=0.5 and 16 OpenMP threads.

| Domain | level | N | Tlist (s) | Teval (s) | Ttotal (s) | Evolume | Ecore |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sphere | 0 | 1,706 | 0.0277 | 0.0388 | 0.0882 | 1.2580e-03 | 6.1289e-03 |
| Sphere | 1 | 13,648 | 0.2624 | 0.6059 | 0.9785 | 2.4490e-04 | 1.3692e-03 |
| Sphere | 2 | 109,184 | 3.2081 | 7.1466 | 11.7366 | 1.8215e-05 | 2.3355e-04 |
| Sphere | 3 | 873,472 | 35.0786 | 75.0498 | 121.2032 | 1.2899e-06 | 2.0342e-06 |
| L-shaped prism | 0 | 867 | 0.0083 | 0.0106 | 0.0317 | 2.6913e-03 | 1.3769e-02 |
| L-shaped prism | 1 | 6,936 | 0.0859 | 0.2162 | 0.3941 | 3.0877e-04 | 3.0114e-03 |
| L-shaped prism | 2 | 55,488 | 1.1955 | 3.1194 | 4.8871 | 8.3921e-05 | 1.3252e-03 |
| L-shaped prism | 3 | 443,904 | 14.6891 | 35.9623 | 56.4175 | 7.7534e-06 | 4.8668e-06 |
| Perforated cube | 0 | 1,049 | 0.0085 | 0.0159 | 0.0397 | 5.1635e-03 | 1.6537e-02 |
| Perforated cube | 1 | 8,392 | 0.1237 | 0.3068 | 0.5164 | 7.0923e-04 | 3.2002e-03 |
| Perforated cube | 2 | 67,136 | 1.6216 | 4.1697 | 6.6407 | 1.3019e-04 | 1.3899e-03 |
| Perforated cube | 3 | 537,088 | 19.0255 | 46.6868 | 72.4079 | 6.6793e-06 | 4.8952e-06 |

Timings were collected on a heavily loaded server and must be
repeated on an idle machine before a performance claim is made.
