# Image Super Resolution Using Deep Convolutional Networks and Upsampling
- Model - SRCNN
- Dataset - BSDS500 (Train and Test), BSDS200 (Validation), Set 5 and 14 (Validation)

### Dataset - Berkeley Segmentation Dataset 500 (BSDS500)
The dataset consists of 500 natural images, ground-truth human annotations and benchmarking code. The data is explicitly separated into disjoint train, validation and test subsets. The dataset is an extension of the BSDS300, where the original 300 images are used for training / validation and 200 fresh images, together with human annotations, are added for testing. Each image was segmented by five different subjects on average.

---
# Super Resolution
- This part of the Repository takes the Input Images and does Super Resolution using SRCNN as the model
- Metric Evaluation is done over 20 epoch trained SRCNN model and Original Images using PSNR and SSIM  
- Found in the `Super Resolution - CNN`  folder
- For more detailed explanation of theory and implementation visit `Super Resolution - CNN/Report.pdf` and `Super Resolution - CNN/demo.mp4`
- The Source files are present in the `Super Resolution - CNN/Code` Folder
- Preprocessing Folder includes functions for Image Distortion, Color Space Conversion, and Metric Evaluation
- The main notebook `Super Resolution - CNN/final.ipynb` contains the whole process

---
# Upsampling
- This part of the Repository upsamples SRCNN output images patches using bicubic interpolation and compares using various similarity metrics 
- Metric Evaluation is done over 9000 epoch trained SRCNN model and Upsampled Images using PSNR, SSIM, RMSE, SRE, SAM, Cosine Similarity
- Found in the `Bicubic Upsampling`  folder
- For more detailed explanation of theory and implementation visit `Bicubic Upsampling/Report.pdf` and `Bicubic Upsampling/demo.mp4`
- The Source files are present in the `Bicubic Upsampling/Code` Folder
- Preprocessing Folder includes functions as previous for Image Distortion, Color Space Conversion
- Postprocessing Folder includes functions for iterative cropping, upsampling, Metric Evaluation and Plotting
- The main notebook `Bicubic Upsampling/final.ipynb` contains the whole process