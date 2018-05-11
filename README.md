# Car-Detection-with-YOLO---CNN
Learn to Use object detection on a car detection dataset and deal with bounding boxes

Run the following cell to load the packages and dependencies that are going to be useful for your journey!

(refer yolo.py)

Important Note: As you can see, we import Keras's backend as K. This means that to use a Keras function in this notebook, you will need to write: K.function(...).

1 - Problem Statement
You are working on a self-driving car. As a critical component of this project, you'd like to first build a car detection system. To collect data, you've mounted a camera to the hood (meaning the front) of the car, which takes pictures of the road ahead every few seconds while you drive around.

You've gathered all these images into a folder and have labelled them by drawing bounding boxes around every car you found. Here's an example of what your bounding boxes look like.

(refer images)

If you have 80 classes that you want YOLO to recognize, you can represent the class label  cc  either as an integer from 1 to 80, or as an 80-dimensional vector (with 80 numbers) one component of which is 1 and the rest of which are 0. The video lectures had used the latter representation; in this notebook, we will use both representations, depending on which is more convenient for a particular step.

In this exercise, you will learn how YOLO works, then apply it to car detection. Because the YOLO model is very computationally expensive to train, we will load pre-trained weights for you to use.

2 - YOLO
YOLO ("you only look once") is a popular algoritm because it achieves high accuracy while also being able to run in real-time. This algorithm "only looks once" at the image in the sense that it requires only one forward propagation pass through the network to make predictions. After non-max suppression, it then outputs recognized objects together with the bounding boxes.

2.1 - Model details
First things to know:

The input is a batch of images of shape (m, 608, 608, 3)
The output is a list of bounding boxes along with the recognized classes. Each bounding box is represented by 6 numbers (pc,bx,by,bh,bw,c)(pc,bx,by,bh,bw,c) as explained above. If you expand cc into an 80-dimensional vector, each bounding box is then represented by 85 numbers.
We will use 5 anchor boxes. So you can think of the YOLO architecture as the following: IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85).

Lets look in greater detail at what this encoding represents.

(refer images)

If the center/midpoint of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Since we are using 5 anchor boxes, each of the 19 x19 cells thus encodes information about 5 boxes. Anchor boxes are defined only by their width and height.

For simplicity, we will flatten the last two last dimensions of the shape (19, 19, 5, 85) encoding. So the output of the Deep CNN is (19, 19, 425).

(refer images)

Now, for each box (of each cell) we will compute the following elementwise product and extract a probability that the box contains a certain class.

(refer images)

Here's one way to visualize what YOLO is predicting on an image:

For each of the 19x19 grid cells, find the maximum of the probability scores (taking a max across both the 5 anchor boxes and across different classes).
Color that grid cell according to what object that grid cell considers the most likely.
Doing this results in this picture:

(refer images)

Note that this visualization isn't a core part of the YOLO algorithm itself for making predictions; it's just a nice way of visualizing an intermediate result of the algorithm.

Another way to visualize YOLO's output is to plot the bounding boxes that it outputs. Doing that results in a visualization like this:

(refer image6)

In the figure above, we plotted only boxes that the model had assigned a high probability to, but this is still too many boxes. You'd like to filter the algorithm's output down to a much smaller number of detected objects. To do so, you'll use non-max suppression. Specifically, you'll carry out these steps:

Get rid of boxes with a low score (meaning, the box is not very confident about detecting a class)
Select only one box when several boxes overlap with each other and detect the same object.

2.2 - Filtering with a threshold on class scores
You are going to apply a first filter by thresholding. You would like to get rid of any box for which the class "score" is less than a chosen threshold.

The model gives you a total of 19x19x5x85 numbers, with each box described by 85 numbers. It'll be convenient to rearrange the (19,19,5,85) (or (19,19,425)) dimensional tensor into the following variables:

box_confidence: tensor of shape  (19×19,5,1)  containing  pcpc  (confidence probability that there's some object) for each of the 5 boxes predicted in each of the 19x19 cells.
boxes: tensor of shape  (19×19,5,4)  containing  (bx,by,bh,bw)  for each of the 5 boxes per cell.
box_class_probs: tensor of shape  (19×19,5,80)  containing the detection probabilities (c1,c2,...c80)  for each of the 80 classes for each of the 5 boxes per cell.

Exercise: Implement yolo_filter_boxes().

Compute box scores by doing the elementwise product as described in Figure 4. The following code may help you choose the right operator:
a = np.random.randn(19*19, 5, 1)
b = np.random.randn(19*19, 5, 80)
c = a * b # shape of c will be (19*19, 5, 80)
For each box, find:
the index of the class with the maximum box score (Hint) (Be careful with what axis you choose; consider using axis=-1)
the corresponding box score (Hint) (Be careful with what axis you choose; consider using axis=-1)
Create a mask by using a threshold. As a reminder: ([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) returns: [False, True, False, False, True]. The mask should be True for the boxes you want to keep.
Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes we don't want. You should be left with just the subset of boxes you want to keep.
Reminder: to call a Keras function, you should use K.function(...).

(refer yolo.py)

Expected Output:

scores[2]	10.7506
boxes[2]	[ 8.42653275 3.27136683 -0.5313437 -4.94137383]
classes[2]	7
scores.shape	(?,)
boxes.shape	(?, 4)
classes.shape	(?,)


2.3 - Non-max suppression
Even after filtering by thresholding over the classes scores, you still end up a lot of overlapping boxes. A second filter for selecting the right boxes is called non-maximum suppression (NMS).

(refer images)

Non-max suppression uses the very important function called "Intersection over Union", or IoU.

(refer images)

Exercise: Implement iou(). Some hints:

In this exercise only, we define a box using its two corners (upper left and lower right): (x1, y1, x2, y2) rather than the midpoint and height/width.
To calculate the area of a rectangle you need to multiply its height (y2 - y1) by its width (x2 - x1).
You'll also need to find the coordinates (xi1, yi1, xi2, yi2) of the intersection of two boxes. Remember that:
xi1 = maximum of the x1 coordinates of the two boxes
yi1 = maximum of the y1 coordinates of the two boxes
xi2 = minimum of the x2 coordinates of the two boxes
yi2 = minimum of the y2 coordinates of the two boxes
In order to compute the intersection area, you need to make sure the height and width of the intersection are positive, otherwise the intersection area should be zero. Use max(height, 0) and max(width, 0).
In this code, we use the convention that (0,0) is the top-left corner of an image, (1,0) is the upper-right corner, and (1,1) the lower-right corner.

(refer yolo.py)

Expected Output:

iou =	0.14285714285714285


You are now ready to implement non-max suppression. The key steps are:

1. Select the box that has the highest score.
2. Compute its overlap with all other boxes, and remove boxes that overlap it more than iou_threshold.
3. Go back to step 1 and iterate until there's no more boxes with a lower score than the current selected box.

This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.

Exercise: Implement yolo_non_max_suppression() using TensorFlow. TensorFlow has two built-in functions that are used to implement non-max suppression (so you don't actually need to use your iou() implementation):

tf.image.non_max_suppression()
K.gather()

(refer yolo.py)

Expected Output:

scores[2]	6.9384
boxes[2]	[-5.299932 3.13798141 4.45036697 0.95942086]
classes[2]	-2.24527
scores.shape	(10,)
boxes.shape	(10, 4)
classes.shape	(10,)


2.4 Wrapping up the filtering
It's time to implement a function taking the output of the deep CNN (the 19x19x5x85 dimensional encoding) and filtering through all the boxes using the functions you've just implemented.

Exercise: Implement yolo_eval() which takes the output of the YOLO encoding and filters the boxes using score threshold and NMS. There's just one last implementational detail you have to know. There're a few ways of representing boxes, such as via their corners or via their midpoint and height/width. YOLO converts between a few such formats at different times, using the following functions (which we have provided):

boxes = yolo_boxes_to_corners(box_xy, box_wh)
which converts the yolo box coordinates (x,y,w,h) to box corners' coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes

boxes = scale_boxes(boxes, image_shape)
YOLO's network was trained to run on 608x608 images. If you are testing this data on a different size image--for example, the car detection dataset had 720x1280 images--this step rescales the boxes so that they can be plotted on top of the original 720x1280 image.

Don't worry about these two functions; we'll show you where they need to be called.

(refer yolo.py)

Expected Output:

scores[2]	138.791
boxes[2]	[ 1292.32971191 -278.52166748 3876.98925781 -835.56494141]
classes[2]	54
scores.shape	(10,)
boxes.shape	(10, 4)
classes.shape	(10,)

**Summary for YOLO**:
- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. 
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
    - Each cell in a 19x19 grid over the input image gives 425 numbers. 
    - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture. 
    - 85 = 5 + 80 where 5 is because $(p_c, b_x, b_y, b_h, b_w)$ has 5 numbers, and and 80 is the number of classes we'd like to detect
- You then select only few boxes based on:
    - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
    - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives you YOLO's final output.

3 - Test YOLO pretrained model on images
In this part, you are going to use a pretrained model and test it on the car detection dataset. As usual, you start by creating a session to start your graph. Run the following cell.

(refer yolo.py)

3.1 - Defining classes, anchors and image shape.
Recall that we are trying to detect 80 classes, and are using 5 anchor boxes. We have gathered the information about the 80 classes and 5 boxes in two files "coco_classes.txt" and "yolo_anchors.txt". Let's load these quantities into the model by running the next cell.

The car detection dataset has 720x1280 images, which we've pre-processed into 608x608 images.

(refer yolo.py)

3.2 - Loading a pretrained model
Training a YOLO model takes a very long time and requires a fairly large dataset of labelled bounding boxes for a large range of target classes. You are going to load an existing pretrained Keras YOLO model stored in "yolo.h5". (These weights come from the official YOLO website, and were converted using a function written by Allan Zelener. References are at the end of this notebook. Technically, these are the parameters from the "YOLOv2" model, but we will more simply refer to it as "YOLO" in this notebook.) Run the cell below to load the model from this file.

(refer yolo.py)

This loads the weights of a trained YOLO model. Here's a summary of the layers your model contains.

(refer yolo.py)

Note: On some computers, you may see a warning message from Keras. Don't worry about it if you do--it is fine.

Reminder: this model converts a preprocessed batch of input images (shape: (m, 608, 608, 3)) into a tensor of shape (m, 19, 19, 5, 85) as explained in Figure (2).

3.3 - Convert output of the model to usable bounding box tensors
The output of yolo_model is a (m, 19, 19, 5, 85) tensor that needs to pass through non-trivial processing and conversion. The following cell does that for you.

(refer yolo.py)

You added yolo_outputs to your graph. This set of 4 tensors is ready to be used as input by your yolo_eval function.

3.4 - Filtering boxes
yolo_outputs gave you all the predicted boxes of yolo_model in the correct format. You're now ready to perform filtering and select only the best boxes. Lets now call yolo_eval, which you had previously implemented, to do this.

(refer yolo.py)

3.5 - Run the graph on an image
Let the fun begin. You have created a (sess) graph that can be summarized as follows:

yolo_model.input is given to yolo_model. The model is used to compute the output yolo_model.output
yolo_model.output is processed by yolo_head. It gives you yolo_outputs
yolo_outputs goes through a filtering function, yolo_eval. It outputs your predictions: scores, boxes, classes
Exercise: Implement predict() which runs the graph to test YOLO on an image. You will need to run a TensorFlow session, to have it compute scores, boxes, classes.

The code below also uses the following function:

image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
which outputs:

image: a python (PIL) representation of your image used for drawing boxes. You won't need to use it.
image_data: a numpy-array representing the image. This will be the input to the CNN.
Important note: when a model uses BatchNorm (as is the case in YOLO), you will need to pass an additional placeholder in the feed_dict {K.learning_phase(): 0}.

(refer yolo.py)

Expected Output:

Found 7 boxes for test.jpg
car	0.60 (925, 285) (1045, 374)
car	0.66 (706, 279) (786, 350)
bus	0.67 (5, 266) (220, 407)
car	0.70 (947, 324) (1280, 705)
car	0.74 (159, 303) (346, 440)
car	0.80 (761, 282) (942, 412)
car	0.89 (367, 300) (745, 648)

The model you've just run is actually able to detect 80 different classes listed in "coco_classes.txt".

What you should remember:

YOLO is a state-of-the-art object detection model that is fast and accurate
It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
You filter through all the boxes using non-max suppression. Specifically:
Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
Intersection over Union (IoU) thresholding to eliminate overlapping boxes
Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.







