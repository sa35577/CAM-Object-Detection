# Class Activation Maps for Object Detection

## Authors:
- Sat Arora (sat.arora@uwaterloo.ca)
- Richard Fan (r43fan@uwaterloo.ca)

This report explores approaches to building a "Class Activation Map" (CAM) around objects. To do this, we train Convolutional Neural Networks (CNNs) on a face/no-face dataset first to classify images, and then, we can generate heat using the weights trained by the CNN.

Sample heatmap and image classifier output (this is also in the report):
<img width="570" alt="image" src="https://github.com/sa35577/CAM-Object-Detection/assets/38817928/cc2457b4-d526-479a-93a6-6fbc17d53ca8">

With the approach to the heatmap being independent of this "binary" classifier, it can be easily extended to training with more than 2 classes (in this case, the classes are face and no-face). Instructions for applying the dataset used in our training process are in the report, and the same directory structure would need to be used for any arbitrary dataset if a user wishes to have minimal code modifications.

The report goes deep into the motivation, theoretical approach, and breaks down the actual code of the whole process. Feel free to take a look at the PDF file for the report, or if you want to play around with the source, the Jupyter Notebook is also attached. Reach out to us if you have any questions!
