# Class Activation Maps for Object Detection

## Authors:
- Sat Arora (sat.arora@uwaterloo.ca)
- Richard Fan (r43fan@uwaterloo.ca)

This report explores approaches to build a "Class Activation Map" around objects. To do this, we train Convolutional Neural Networks (CNNs) on a face/no-face dataset first to classify images, and then, we can generate heat using the weights trained by the CNN.

The report goes deep into the motivation, theoretical approach, and breaks down the actual code of the whole process. Feel free to take a look at the PDF file for the report, or if you want to play around with the source, the Jupyter Notebook is also attached. Reach out to us if you have any questions!

Sample heatmap and image classifier output (this is also in the report):
<img width="591" alt="image" src="https://github.com/sa35577/CAM-Object-Detection/assets/38817928/8ecc6ce6-dc6f-47d9-a49b-7e925c6975ca">
