### Neural Net PSO ###

This program uses a hybrid of PSO and a standard Perceptron to classify digits that are in matrix form.

To add a set of test and train files, add them in to /files folder, and title them with the following convention:
train{image_size}.txt, test{image_size}.txt.

We currently have to train/test sets in the folder. One 8x8 image set and one 32x32 image. 

To run the program cd into the root directory and run the following command:

$python3 Classifier.py {learning rate} {num epochs} {image size} {num output nodes} {num PSO iterations} {Seed Radius} {Seed Velocity}

Sample Command:

$python3 Classifier.py .1 30 8 1 10 .05 .15

Learning Rate: a value for the Perceptron learning rate. Typical values of .01 for 1 output node and .1 for 10 output nodes.
Num epochs: The number of training epochs to run. Typical values between 10 and 50.
Image Size= The size of the image set (8 or 32)
Num output nodes= Number of Perceptron output nodes ( 1 or 10, for best results use 10).
Num PSO iterations= Number of PSO iteration to run on a given epoch.
Seed Velocity= Particle Seed Velocity
Seed Radius= Particle seed radius

The number of particles and the neighborhood topology are fixed in the code in order to minimize the number of command line inputs. To change these 
refer to the PSO and Neighborhood Class respectivley and change the init variables. 
