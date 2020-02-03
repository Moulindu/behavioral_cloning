import numpy as np
import os
import csv
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, Activation, Dropout
import cv2
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split





samples = [] 

with open('./data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None) 
    for line in reader:
        samples.append(line)

        

print("done")

train_samples, validation_samples = train_test_split(samples,test_size=0.2)



#building the generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: #Loop forever so that the generator never terminates
        shuffle(samples) 
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3): # taking 3 images - centre, left and right
                        
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) # converting it to RGB 
                        center_angle = float(batch_sample[3]) # getting the steering angle 
                        images.append(center_image)
                        
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2) #increasing the steering angle by 0.2 for left images
                        elif(i==2):
                            angles.append(center_angle-0.2) #decreasing the steering angle by 0.2 for right images
                        
                        # performing data augmentation by flipping the image and inversing the sign
                        # of the corresponding steering angle
                        
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                          
                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) 
            

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



# Building the model 
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # Normalization
model.add(Cropping2D(cropping=((60,25),(0,0)))) #trimming image to focus on the road portion   




# three layers of Convolution of channel sizes 24, 36 and 48, filter size= 5x5, stride= 2x2 as per the suggestions
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('relu'))

model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('relu'))




# convolution with no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu')) # using elu to reduce vanishing gradient

model.add(Flatten())

# fully connected layers [100, 50, 10, 1] as per the suggestions

# fully connected layer 
model.add(Dense(100))
model.add(Activation('relu'))

# adding a dropout layer with keep probability 0.5 to reduce overfitting
model.add(Dropout(0.5))

# fully connected layer 
model.add(Dense(50))
model.add(Activation('relu'))

# fully connected layer 
model.add(Dense(10))
model.add(Activation('relu'))

# fully connected layer 
model.add(Dense(1)) 


# using mean square error and adam optimizer
model.compile(loss='mse',optimizer='adam')


# training the model with fit_generator
# no of epochs : 5

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

#saving the model
model.save('model2.h5')

print(' Model saved ..')

# keras method to print the model summary
model.summary()
