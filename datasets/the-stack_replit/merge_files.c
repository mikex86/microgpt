#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>

#define BUFFER_SIZE 1073741824

void mergeFiles(const char* folderPath, const char* outputFile) {
    FILE *out = fopen(outputFile, "wb");

    if (out == NULL) {
        printf("Failed to open output file: %s\n", outputFile);
        return;
    }

    DIR *dir;
    struct dirent *ent;

    dir = opendir(folderPath);

    unsigned char *buffer = malloc(BUFFER_SIZE);

    if (dir != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            char filePath[PATH_MAX];
            snprintf(filePath, sizeof(filePath), "%s/%s", folderPath, ent->d_name);

            struct stat fileStat;
            if (stat(filePath, &fileStat) == 0 && S_ISREG(fileStat.st_mode)) {  // Process regular files only

                printf("Processing file: %s\n", filePath);
                FILE *in = fopen(filePath, "rb");

                if (in != NULL) {
                    size_t bytesRead;

                    while ((bytesRead = fread(buffer, sizeof(unsigned char), BUFFER_SIZE, in)) > 0) {
                        fwrite(buffer, sizeof(unsigned char), bytesRead, out);
                    }

                    fclose(in);
                } else {
                    printf("Failed to open input file: %s\n", filePath);
                }
            }
        }
        closedir(dir);
    } else {
        printf("Failed to open folder: %s\n", folderPath);
    }

    free(buffer);

    fclose(out);
}

int main() {
    const char* folderPath = "files";
    const char* outputFile = "merged.bin";

    printf("Merging files from folder: %s\n", folderPath);

    mergeFiles(folderPath, outputFile);

    return 0;
}
