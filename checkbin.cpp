#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fin = fopen("results.bin", "rb");
    if (!fin) return 1;
    double row[4];
    int row_num = 0;
    while (fread(row, sizeof(double), 4, fin) == 4) {
        row_num++;
        if (row[3] < 0.0 ) {
            printf("Row %d: %e %e %e %e\n", row_num, row[0], row[1], row[2], row[3]);
        }
    }
    fclose(fin);
    return 0;
}