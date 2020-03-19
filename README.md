# ImageSegmentation
Image Segmentation with different algorithms: including karger, boykov kolmogorov, pymaxflow and push relabel.

The main components of the project are the following: 
- IMAGES is folder with a selection of images, downloaded form internet
- GUI processes the image and enables to integrate the scribbles
- IMAGE PREPROCESSING calcultates the weights of the graph created from the image and the scribbles
- GRAPHCUT contains the different maxflow mincut algorithms, implemented from scratch; as well as the Superpixel file that transforms an image before being processed by the Karger algorihtm
