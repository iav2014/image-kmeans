# Using kmeans in  images

Theses python scripts are used to preprocess image using kmeans
and using for area calculations of satellite images, change backgrounds of images, 

pip install packages:
 - numpy 
 - plt
 - PIL 
 - sklearn 
 - argparse

example of area calculation:
`python image_segmentation.py --file leman_lake_1_3364.5.jpg --area 3364`

example of change background color:
you can change centroids number and test with your images
`python kmeans_image.py --file 1.png`

example of random pixels:
`python centroides.py`

Nacho Ariza, Dec 2020
MIT License

