/*  SeamCarver - Content-aware image resizing
    Author: Stavros Kladis <stavroskladis@hotmail.com>
    Copyright © 2025 Stavros Kladis
*/
#pragma once

#include <memory>
#include <vector>

/**
 * @brief Computes the energy map of an image using Sobel gradient operators
 *
 * This function calculates the energy map for seam carving by computing the magnitude
 * of image gradients using Sobel operators. High energy pixels correspond to edges
 * and important image features that should be preserved during seam carving.
 *
 * @param pixels Pointer to the input image pixel data in RGB format
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param channels Number of color channels (default: 3 for RGB)
 * @return Returns nullptr if the computation fails (invalid parameters or unsupported format).
 */
std::unique_ptr<std::vector<int>> energyMapCompute(const unsigned char *pixels, int width,
                                                   int height, int channels = 3);

/**
 * @brief Custom energy map computation for demonstration purposes (used only in debug builds)
 *
 * This function implements the complete energy map computation step by step,
 * to demonstrate exactly what the OpenCV version does under the hood. This is useful
 * for understanding the algorithm and comparing performance with the optimized OpenCV version.
 *
 * The function performs:
 * 1. RGB to grayscale conversion
 * 2. Sobel gradient computation (X and Y directions)
 * 3. Energy map calculation (|Gx| + |Gy|)
 * 4. Border handling
 *
 * @param pixels Pointer to the input image pixel data in RGB format
 * @param width Width of the image in pixels
 * @param height Height of the image in pixels
 * @param channels Number of color channels (default: 3 for RGB)
 * @return Returns nullptr if the computation fails (invalid parameters or unsupported format).
 */
std::unique_ptr<std::vector<int>> energyMapComputeCustom(const unsigned char *pixels, int width,
                                                         int height, int channels = 3);

/**
 * @brief Calculates the energy of a single using Sobel operators
 *
 * This function computes the energy value for a single pixel by applying Sobel
 * gradient operators manually. Used for incremental energy map updates.
 *
 * Because it operates on a single coordinate, it is cheap enough to be called thousands of
 * times per frame, letting us update only the affected region after each seam removal.
 *
 * @param pixels Pointer to the input image pixel data
 * @param x X-coordinate of the pixel
 * @param y Y-coordinate of the pixel
 * @param width Width of the image
 * @param height Height of the image
 * @param channels Number of color channels
 * @return Energy value for the specified pixel
 */
int calculateSinglePixelEnergy(const unsigned char *pixels, int x, int y, int width, int height,
                               int channels);

/**
 * @brief Incrementally updates energy map around a removed seam
 *
 * This function efficiently updates only the energy values around a removed seam,
 * avoiding the need to recalculate the entire energy map. This provides significant
 * performance improvements for iterative seam carving.
 *
 * @param energy Reference to the energy map vector to update
 * @param seam Vector containing the x-coordinates of the removed seam
 * @param width Current image width
 * @param height Image height
 * @param pixels Pointer to the current image pixel data
 * @param channels Number of color channels
 */
void updateEnergyMapAroundSeam(std::vector<int> &energy, const std::vector<int> &seam, int width,
                               int height, const unsigned char *pixels, int channels);

/**
 * @brief Removes the same vertical seam from an energy map (in-place).
 *
 * After removing a seam from the pixel buffer (width shrinks by 1), the energy map must also
 * shrink by 1 column per row. If we don't, subsequent seam searches will interpret the energy
 * buffer with the wrong row stride, causing seams to drift/bias (often to one side).
 *
 * @param energy    Energy map in row-major order (size must be old_width * height)
 * @param old_width Width before seam removal
 * @param height    Image height
 * @param seam      Seam x-coordinates in the OLD width (one per row)
 * @return true on success, false on invalid input
 */
bool removeSeamFromEnergy(std::vector<int> &energy, int old_width, int height,
                          const std::vector<int> &seam);

/**
 * @brief Finds the lowest energy vertical seam in an image using a greedy algorithm.
 *
 * This function implements a greedy approach to seam carving by finding a connected
 * path of pixels from the bottom to the top of the image with minimal cumulative energy.
 * The algorithm makes local optimal choices at each step, which may not always result
 * in the globally optimal seam but is fast and produces good results for most images.
 *
 * @param energy A unique_ptr to the energy matrix. The function takes ownership
 *               of this data and the original pointer becomes invalid after the call.
 *               The energy matrix should be in row-major order: energy[y * width + x].
 * @param width The width of the image (number of columns in the energy matrix).
 * @param height The height of the image (number of rows in the energy matrix).
 * @return A vector of length height where seam[y] contains the x-coordinate of the
 *         seam pixel in row y. The seam is guaranteed to be 4-connected (each pixel
 *         is adjacent to the seam pixel in the next row).
 */
void findLowestEnergySeam(std::unique_ptr<std::vector<int>> energy, std::vector<int> &seam_out,
                          int width, int height);

/**
 * @brief Finds the lowest energy vertical seam using Dynamic Programming for global optimality.
 *
 * This function implements a Dynamic Programming approach to seam carving that finds
 * the globally optimal seam by considering all possible paths from bottom to top.
 * Unlike the greedy approach, this algorithm guarantees the minimum cumulative energy
 * seam but requires O(w*h) time and space complexity.
 *
 * The algorithm works by:
 * 1. Building a cumulative energy matrix from bottom to top
 * 2. For each pixel, considering the minimum energy path from the three possible
 *    parent pixels in the row below (left, center, right)
 * 3. Backtracking from the minimum energy pixel in the top row to find the optimal seam
 *
 * @param energy A unique_ptr to the energy matrix. The function takes ownership
 *               of this data and the original pointer becomes invalid after the call.
 *               The energy matrix should be in row-major order: energy[y * width + x].
 * @param seam_out Reference to output vector where seam[y] will contain the x-coordinate
 *                 of the seam pixel in row y.
 * @param width The width of the image (number of columns in the energy matrix).
 * @param height The height of the image (number of rows in the energy matrix).
 */
void findLowestEnergySeamDP(std::unique_ptr<std::vector<int>> energy, std::vector<int> &seam_out,
                            int width, int height);

/**
 * @brief Removes a vertical seam from the image data
 *
 * This function physically removes a vertical seam from the image by shifting
 * pixels to the left to fill the gap created by removing the seam pixels.
 * The image width is reduced by 1 pixel after each seam removal.
 *
 * @param pixels Input/output pixel data in RGB format. The function modifies
 *               this buffer in-place to remove the seam.
 * @param width Current image width (will be decremented by 1 on successful removal)
 * @param height Image height (unchanged)
 * @param seam Vector containing x-coordinates of pixels to remove, one per row.
 *             seam[y] gives the x-coordinate of the pixel to remove in row y.
 * @return true if the seam was successfully removed, false otherwise
 */
bool removeSeam(unsigned char *pixels, int &width, int height, const std::vector<int> &seam);

/**
 * @brief Builds a horizontally-resized copy of `src` using simple bilinear interpolation.
 * @param src            Pointer to source RGB data (8-bit per channel)
 * @param width, height  Dimensions of the source image
 * @param channels       Number of colour channels (expected 3)
 * @param target_width   Desired width (height remains height)
 * @return               Unique_ptr to resized pixel data (nullptr on failure)
 */
std::unique_ptr<unsigned char[]> buildPrimitiveResizeTexture(const unsigned char *src, int width,
                                                             int height, int channels,
                                                             int target_width);