/*  SeamCarver - Content-aware image resizing
    Author: Stavros Kladis <stavroskladis@hotmail.com>
    Copyright © 2025 Stavros Kladis
*/
#define _CRT_SECURE_NO_WARNINGS
#include <algorithm> // for std::clamp
#include <chrono>    // for timing
#include <climits>   // for INT_MAX
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "image_processing.hpp"

// Sobel kernels for manual computation
const int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

const int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Convert RGB to grayscale (0.299*Red + 0.587*Green + 0.114*Blue)
unsigned char rgbToGray(unsigned char r, unsigned char g, unsigned char b) {
    return static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

// Calculate energy for a single pixel manually for the center pixel (x,y) of the 3x3 neighborhood.
int calculateSinglePixelEnergy(const unsigned char *pixels, int x, int y, int width, int height,
                               int channels) {

#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    int gx = 0, gy = 0;

    // Apply Sobel kernels for this pixel (the order of the loops does matter for performance!).
    // Memory access pattern: Images are stored in row-major order (row by row)
    // Cache efficiency: Accessing pixels in row order is more cache-friendly
    // Performance: The current ky (rows) outer, kx (columns) inner matches the memory layout
    for (int ky = -1; ky <= 1; ky++) { // y-direction (rows)

        for (int kx = -1; kx <= 1; kx++) { // x-direction (columns)

            // Calculating the coordinates of each neighbor around the center pixel (x,y)
            int nx = x + kx, ny = y + ky;

            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                // Clamping border pixels to ensure every pixel has a full 3x3 neighborhood
                // This gives more stable results along image edges
                nx = std::clamp(nx, 0, width - 1);  // clamp to 0 or width-1
                ny = std::clamp(ny, 0, height - 1); // clamp to 0 or height-1
            }

            // Convert RGB to grayscale
            // Image is stored as: [R1,G1,B1, R2,G2,B2, R3,G3,B3, ...]
            // This calculates the memory index for a pixel at coordinates (nx,ny):
            // ny * width: Skip to the correct row
            // + nx: Move to the correct column within that row
            // Multiply by 3 (RGB channels) to get the byte index
            int idx = (ny * width + nx) * channels;
            unsigned char gray = rgbToGray(pixels[idx], pixels[idx + 1], pixels[idx + 2]);

            // Apply Sobel kernels for this pixel
            gx += gray * sobelX[ky + 1][kx + 1];
            gy += gray * sobelY[ky + 1][kx + 1];
        }
    }

    // Calculate the energy for this pixel
    int result = abs(gx) + abs(gy);

#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    static long long total_time = 0;
    static int pixel_count = 0;
    total_time += duration.count();
    pixel_count++;

    // Log average every 100 pixels to avoid spam
    if (pixel_count % 100 == 0) {
        spdlog::debug("calculateSinglePixelEnergy: average time per pixel: {} ns ({} pixels)",
                      total_time / pixel_count, pixel_count);
    }
#endif

    return result;
}

// Incremental energy map update around removed seam (instead of recalculating the whole energy map)
void updateEnergyMapAroundSeam(std::vector<int> &energy, const std::vector<int> &seam, int width,
                               int height, const unsigned char *pixels, int channels) {

    // height is the number of rows (y-direction) and it hasn't changed - only the width shrinks
    // so, for each row, we update the energy map around the seam
    for (int y = 0; y < height; y++) {

        int seam_x = seam[y]; // seam vector holds the x-coordinates (column indices) of the seam
                              // pixel in each row, as produced by findLowestEnergySeam()

        // Only update 3 pixels per row (around the seam): [seam_x - 1 , seam_x , seam_x + 1]
        // std::max(0, seam_x - 1) clamps at the left border (just reuse the closest border pixel)
        // std::min(width - 1, seam_x + 1) clamps at the right border.
        for (int x = std::max(0, seam_x - 1); x <= std::min(width - 1, seam_x + 1); x++) {
            // After seam removal, index seam_x now contains a real pixel (it used to be seam_x+1),
            // so we must recompute it too.
            energy[y * width + x] =
                calculateSinglePixelEnergy(pixels, x, y, width, height, channels);
        }
    }
}

bool removeSeamFromEnergy(std::vector<int> &energy, int old_width, int height,
                          const std::vector<int> &seam) {
    if (old_width <= 1 || height <= 0 || seam.size() != static_cast<size_t>(height)) {
        return false;
    }
    if (energy.size() != static_cast<size_t>(old_width) * static_cast<size_t>(height)) {
        return false;
    }

    const int new_width = old_width - 1;

    // Compact each row from old_width -> new_width, skipping the seam element.
    for (int y = 0; y < height; y++) {
        const int seam_x = seam[y];
        if (seam_x < 0 || seam_x >= old_width) {
            return false;
        }

        int *src_row = energy.data() + static_cast<size_t>(y) * old_width;
        int *dst_row = energy.data() + static_cast<size_t>(y) * new_width;

        // Copy left side (0..seam_x-1)
        if (seam_x > 0) {
            memmove(dst_row, src_row, static_cast<size_t>(seam_x) * sizeof(int));
        }

        // Copy right side (seam_x+1..old_width-1) into dst starting at seam_x
        const int right_count = old_width - seam_x - 1;
        if (right_count > 0) {
            memmove(dst_row + seam_x, src_row + seam_x + 1,
                    static_cast<size_t>(right_count) * sizeof(int));
        }
    }

    energy.resize(static_cast<size_t>(new_width) * static_cast<size_t>(height));
    return true;
}

// Custom energy map computation (for demonstration purposes) - to compare with OpenCV
// implementation
std::unique_ptr<std::vector<int>> energyMapComputeCustom(const unsigned char *pixels, int width,
                                                         int height, int channels) {

#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    // Step 1: Convert RGB to grayscale (for every pixel of the image!)
    std::vector<unsigned char> gray(width * height);
    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;
        gray[i] = rgbToGray(pixels[idx], pixels[idx + 1], pixels[idx + 2]);
    }

#ifdef DEBUG
    auto gray_end = std::chrono::high_resolution_clock::now();
    auto gray_duration = std::chrono::duration_cast<std::chrono::microseconds>(gray_end - start);
    spdlog::debug("Custom energyMapCompute: Grayscale conversion took {} μs",
                  gray_duration.count());
#endif

    // Step 2: Compute Sobel gradients manually
    std::vector<int> gx(width * height);
    std::vector<int> gy(width * height);

    // We skip the border pixels (first and last row/column) since
    // the loop starts at 1 and ends at width-2 and height-2 respectively
    // Instead of clamping the border pixels, we skip them and we fill the border pixels later
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx_val = 0, gy_val = 0;

            // Apply Sobel X kernel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    // (y + ky) * width + (x + kx) calculates the memory index
                    int pixel = gray[(y + ky) * width + (x + kx)];
                    gx_val += pixel * sobelX[ky + 1][kx + 1];
                    gy_val += pixel * sobelY[ky + 1][kx + 1];
                }
            }

            gx[y * width + x] = gx_val;
            gy[y * width + x] = gy_val;
        }
    }

#ifdef DEBUG
    auto sobel_end = std::chrono::high_resolution_clock::now();
    auto sobel_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(sobel_end - gray_end);
    spdlog::debug("Custom energyMapCompute: Sobel computation took {} μs", sobel_duration.count());
#endif

    // Step 3: Compute the energy map (|Gx| + |Gy|)
    std::vector<int> energy(width * height);

    // Iterates the full map once again (not efficient, but we do this for demonstration purposes)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            energy[y * width + x] = abs(gx[y * width + x]) + abs(gy[y * width + x]);
        }
    }

    // Handle borders (copy edge pixels)
    for (int y = 0; y < height; y++) {
        energy[y * width] = energy[y * width + 1];                     // left edge
        energy[y * width + width - 1] = energy[y * width + width - 2]; // right edge
    }
    for (int x = 0; x < width; x++) {
        energy[x] = energy[width + x];                                       // top edge
        energy[(height - 1) * width + x] = energy[(height - 2) * width + x]; // bottom edge
    }

#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto energy_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - sobel_end);
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    spdlog::debug("Custom energyMapCompute: Energy computation took {} μs",
                  energy_duration.count());
    spdlog::info("Custom energyMapCompute: {}x{} image took {} μs (vs OpenCV: ~18000 μs)", width,
                 height, total_duration.count());
#endif

    return std::make_unique<std::vector<int>>(energy);
}

// OpenCV implementation of the energy map computation (optimal solution).
// This is the implementation that is used in the final product (release builds).
std::unique_ptr<std::vector<int>> energyMapCompute(const unsigned char *pixels, int width,
                                                   int height, int channels) {

#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    // Use cv_type and the switch for possible future use instead of harcoding RGB (3 channels)
    int cv_type;
    switch (channels) {
    case 1:
        cv_type = CV_8UC1; // Grayscale
        spdlog::error("Grayscale images are not supported");
        break;
    case 3:
        cv_type = CV_8UC3; // RGB
        break;
    case 4:
        cv_type = CV_8UC4; // RGBA
        spdlog::error(
            "We don't support RGBA since it's unnecessary complexity for camera streams!");
        break;
    default:
        return nullptr; // Error
    }
    if (cv_type != CV_8UC3) {
        return nullptr;
    } // We support only RGB for now

    // wrap stb_image buffer in a cv::Mat header (no copy)
    // CV_8UC3 -> 3-channels x 8 bit unsigned each = 24bits/pixel (typical RGB frame)
    cv::Mat src(height, width, cv_type, const_cast<unsigned char *>(pixels));

    // Convert to 1 channel grayscale (1 byte per pixel)
    // Grayscale = 0.299*Red + 0.587*Green + 0.114*Blue
    // For more info: https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
    cv::Mat dst;
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);

    cv::Mat gx; // will contain horizontal gradient (how much brightness changes left-to-right)
    cv::Mat gy; // will contain vertical gradient (how much brightness changes top-to-bottom)

    // Sobel is applied to the single grayscale channel (dst)
    // This computes how much brightness changes in horizontal (gx) and vertical (gy) directions

    // Compute gradient in X direction (horizontal)
    cv::Sobel(dst, gx, CV_16S, 1, 0); // dx=1, dy=0 (horizontal edges)

    // Compute gradient in Y direction (vertical)
    cv::Sobel(dst, gy, CV_16S, 0, 1); // dx=0, dy=1 (vertical edges)

    cv::Mat abs_gx, abs_gy;
    cv::convertScaleAbs(gx, abs_gx); // |gx|
    cv::convertScaleAbs(gy, abs_gy); // |gy|

    // Total Energy at each pixel = |horizontal_gradient| + |vertical_gradient|
    // tells us how "important" that pixel is for preserving image structure.
    // High energy = big brightness change = important edge to keep (it preserves image structure)
    // Low energy = smooth area = good place to carve a seam (good candidate for removal)
    cv::Mat energy;
    cv::add(abs_gx, abs_gy, energy); // E = |gx| + |gy|

    // Copy the raw pixel data from the cv::Mat into a std::vector<int>
    auto result = std::make_unique<std::vector<int>>(energy.begin<uchar>(), energy.end<uchar>());

#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    spdlog::debug("energyMapCompute: {}x{} image took {} μs", width, height, duration.count());
#endif

    return result;
}

// Greedy approach to find the lowest energy seam
void findLowestEnergySeam(std::unique_ptr<std::vector<int>> energy, std::vector<int> &seam_out,
                          int width, int height) {

#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    // Use pass by ref for seam to avoid repeated allocations due to the returning of the vector in
    // seam carving loops (this function will be called hundreds of times per image resize)
    // Also, if we copied 4KB every frame at: 30 FPS = 120KB/second
    // Memory allocation at every frame = potential fragmentation
    // CPU cycles for copying = reduced performance
    // Cache pollution = affects other real-time tasks
    // Battery life - unnecessary CPU work drains power
    // auto seam_out = std::make_unique<std::vector<int>>(height);

    // The vector seam_out is pre-allocated by the caller and passed by reference,
    // so this function incurs zero allocations/deallocations per call.

    // Start from the bottom row, find minimum energy pixel
    int min_energy = INT_MAX;
    int start_x = 0;
    int y = height - 1; // 0 is geometrically the top row of the image, height - 1 is the bottom row

    // Find starting position (row-major order, so the image bottom row is the last row: height - 1)
    // Complexity: O(width) - one comparison per column (we have width number of columns)
    for (int x = 0; x < width; x++) {

        // index = y rows above x columns (places us to the desired row - in this case the last)
        //          + offset (places us to the x-th column of the desired row)
        int pixel_energy = (*energy)[y * width + x];

        // Scans the row from left to right, and finds the pixel with the minimum energy
        if (pixel_energy < min_energy) {
            min_energy = pixel_energy;
            start_x = x;
        }
    }

    seam_out[height - 1] = start_x; // Save the index of the bottom pixel with the minimum energy

    // Trace upward using greedy approach (locally optimal). It's extremely fast and cache-friendly.
    // We never reconsider earlier picks or keep alternative paths.
    // In each of these rows the algorithm already knows the seam's x-coordinate in the row below
    // (current_x). It has to inspect the three possible successors directly above that pixel:
    // [current_x-1, current_x, current_x+1] It chooses the smallest of those three and stores it in
    // seam_out[y]. Complexity: O(width) - one comparison per column (we have width number of
    // columns)
    for (y = height - 2; y >= 0; y--) {

        int current_x = seam_out[y + 1]; // position from row below
        int best_x = current_x; // assume we will stay in the same column (as a "best" choice)
        int min_energy = (*energy)[y * width + current_x];

        // If we are not at the left edge
        if (current_x > 0) {

            int left_energy = (*energy)[y * width + (current_x - 1)];

            // Check the left neighbour
            if (left_energy < min_energy) {
                min_energy = left_energy;
                best_x = current_x - 1;
            }
        }

        // If we are not at the right edge
        if (current_x < width - 1) {

            int right_energy = (*energy)[y * width + (current_x + 1)];

            // Check the right neighbour
            if (right_energy < min_energy) {
                min_energy = right_energy; // store it for consistency, not needed to be stored here
                best_x = current_x + 1;
            }
        }

        seam_out[y] = best_x;
    }

#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    spdlog::debug("findLowestEnergySeam: {}x{} image took {} μs", width, height, duration.count());
#endif

    // Returning by copy 120KB/second is not a good idea
    // Using move semantics is by far better (we transfer the onwership)
    // but, not optimal due to the allocations / deallocations in each call of the method if we
    // do the allocation inside it! Passing by ref a prealocated vector is the optimal choice here.
}

// In contrast to the greedy approach, the dynamic-programming version keeps a cumulative table of
// all possible paths and back-tracks to get the global optimum.
// Complexity: O(width x height) - we have to fill the table for each pixel in the image.
void findLowestEnergySeamDP(std::unique_ptr<std::vector<int>> energy, std::vector<int> &seam_out,
                            int width, int height) {

#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    // Create cumulative energy matrix (dp table)
    std::vector<int> dp(width * height);

    // Avoid recalculation inside the loop (we could had applied this optimisation in other places)
    const int bottomOffset = (height - 1) * width;

    // Initialize bottom row with original energy values (we need to modify them, thus we copy)
    for (int x = 0; x < width; x++) {
        dp[bottomOffset + x] = (*energy)[bottomOffset + x];
    }

    // Fill dp table from bottom to top (the last row (height-1) is already initialised)
    for (int y = height - 2; y >= 0; y--) { // When we compute a row, every row below it is finished

        for (int x = 0; x < width; x++) { // for every pixel of the row

            int current_energy = (*energy)[y * width + x];
            int min_parent_energy = INT_MAX;

            // Check three possible parent pixels from row below and
            // choose the parent with the minimum energy
            for (int dx = -1; dx <= 1; dx++) {

                int parent_x = x + dx;

                if (parent_x >= 0 && parent_x < width) { // bounds check

                    int parent_energy = dp[(y + 1) * width + parent_x];

                    if (parent_energy < min_parent_energy) {
                        min_parent_energy = parent_energy;
                    }
                }
            }

            // Add to the current energy of the pixel, the energy of the parent with the minimum
            // energy
            dp[y * width + x] = current_energy + min_parent_energy;
        }
    }

    // Find starting position (minimum energy in top row)
    // We start now the backtracking process (from the top row to the bottom row)
    int min_energy = INT_MAX;
    int start_x = 0;

    // Scan every column in the top row and find the cheapest path to the bottom row
    for (int x = 0; x < width; x++) {
        if (dp[x] < min_energy) {
            min_energy = dp[x];
            start_x = x;
        }
    }

    seam_out[0] = start_x; // top pixel with minimum energy

    // Backtrack (top row 1 -> bottom row height-1) to find the optimal seam
    for (int y = 1; y < height; y++) {

        int current_x = seam_out[y - 1]; // position from row above
        int best_x = current_x;
        int min_parent_energy = INT_MAX;

        // Check three possible parent pixels from row below
        for (int dx = -1; dx <= 1; dx++) {

            int parent_x = current_x + dx;
            if (parent_x >= 0 && parent_x < width) {

                int parent_energy = dp[y * width + parent_x];
                if (parent_energy < min_parent_energy) {
                    min_parent_energy = parent_energy;
                    best_x = parent_x;
                }
            }
        }

        seam_out[y] = best_x;
    }

#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    spdlog::debug("findLowestEnergySeamDP: {}x{} image took {} μs", width, height,
                  duration.count());
#endif
}

// Removes a seam from the image. This approach is memory-efficient because it modifies the buffer
// in-place without requiring any additional memory allocation. The function modifies the width of
// the image by 1, and returns true if the seam is valid, false otherwise.
bool removeSeam(unsigned char *pixels, int &width, int height, const std::vector<int> &seam) {

#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    if (!pixels || width <= 1 || height <= 0 || seam.size() != static_cast<size_t>(height)) {
        return false;
    }

    const int channels = 3; // RGB format
    const int old_width = width;
    width--; // Reduce width by 1 (new width after removal)

    // Start from the top row and go down to the bottom row
    for (int y = 0; y < height; y++) {

        int seam_x = seam[y];
        if (seam_x < 0 || seam_x >= old_width) {
            return false; // If the seam we retrieved is out of bounds, we return false
        }

        // Calculate pointers to the beginning of the current row in src/dst layout
        // Points to the original row data (width = old_width), before seam removal
        // Multiply by channels to convert the pixel index to byte index
        unsigned char *src_row = pixels + (static_cast<size_t>(y) * old_width) * channels;

        // Points to the destination row (width = width - 1), after seam removal
        unsigned char *dst_row = pixels + (static_cast<size_t>(y) * width) * channels;

        // Copy pixels left of the seam (0 .. seam_x-1)
        if (seam_x > 0) {
            // if the pixel is not left edged, copy: numBytes = seam_x * channels (bytes)
            memmove(dst_row, src_row, static_cast<size_t>(seam_x) * channels);
        }

        // Copy pixels right of the seam (seam_x+1 .. old_width-1). The seam pixel gets overwritten
        // Number of pixels to the right of seam (-1 becase we exclude the seam pixel)
        int right_count = old_width - seam_x - 1;
        if (right_count > 0) {
            memmove(dst_row + seam_x * channels,       // destination starts at seam_x position
                    src_row + (seam_x + 1) * channels, // source skips the seam pixel
                    static_cast<size_t>(right_count) * channels);
        }
    }

#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    spdlog::debug("removeSeam: {}x{} image took {} μs", width + 1, height, duration.count());
#endif

    return true;
}

// This function implements bilinear interpolation for horizontal image resizing.
// It resizes an image from width to target_width, maintaining the same height.
std::unique_ptr<unsigned char[]> buildPrimitiveResizeTexture(const unsigned char *src, int width,
                                                             int height, int channels,
                                                             int target_width) {

#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();
#endif

    if (!src || target_width <= 0 || width <= 0 || height <= 0 || channels <= 0)
        return nullptr;

    // Number of pixels in the destination image
    const size_t pixel_count = static_cast<size_t>(target_width) * height * channels;

    // This will hold the resized image.
    std::unique_ptr<unsigned char[]> dst = std::make_unique<unsigned char[]>(pixel_count);

    // Iterate over each row of the destination image
    for (int y = 0; y < height; ++y) {

        // Iterate over each pixel (column) of the current row of the destination image (x)
        for (int x = 0; x < target_width; ++x) {

            // Use the ratio of source width to target width to determine where each destination
            // pixel "lands" in the source image, allowing for proportional scaling.
            // src_x represents the exact position in the source image that maps to the current
            // destination pixel x.
            float src_x = static_cast<float>(x) * width / target_width;

            // Find the two nearest source pixels (x1 and x2) for interpolation
            int x1 = static_cast<int>(src_x);     // Left source pixel (integer part of src_x)
            int x2 = std::min(x1 + 1, width - 1); // Right (upper) source pixel
                                                  // (next pixel, clamped to image width)

            // Weight for the interpolation (the fractional part of src_x).
            // Expresses how much to blend between the two source pixels.
            // If src_x = 3.7, x1 = 3 => x2 = 4 and w = 0.7, thus we blend 70% of
            // the left pixel and 30% of the right pixel. Results in a smooth transition
            // between the two pixels.
            float w = src_x - x1;

            // Iterate over each channel of the pixel (interpolate each channel independently)
            for (int c = 0; c < channels; ++c) {

                // Calculate the indices for the source pixels (row-major order)
                // y * width: Skip to the correct row in the source image (could be precomputed)
                int index1 = (y * width + x1) * channels + c; // Left source pixel
                int index2 = (y * width + x2) * channels + c; // Right source pixel

                // y * target_width: Skip to the correct row in the destination image
                // * channels: Convert from pixel number to value index (each pixel has 3 values)
                int id = (y * target_width + x) * channels + c; // Destination pixel (result)

                // Get the values of the source pixels (the actual pixel values to interpolate with)
                unsigned char v1 = src[index1]; // Left source pixel value (8-bit values (0-255))
                unsigned char v2 = src[index2]; // Right source pixel value (0=black to 255=white)

                // Interpolate the value of the destination pixel using the interpolation formula.
                // v1 * (1.0f - w): Contribution from left source pixel
                // v2 * w: Contribution from right source pixel
                // dst[id] is a weighted average of the two source pixels.
                dst[id] = static_cast<unsigned char>(v1 * (1.0f - w) + v2 * w);
            }
        }
    }

#ifdef DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    spdlog::debug("buildPrimitiveResizeTexture: {}x{} -> {}x{} took {} μs", width, height,
                  target_width, height, duration.count());
#endif

    return dst;
}