"""
This module is sample code that shows how to invoke the web service to get datasets
This example is meant for processing fashion MNIST data
Author: Palacode Narayana Iyer Anantharaman
Date: 18 Oct 2018
"""
import numpy as np
import matplotlib.pyplot as plt
from api_client import ApiClient

# Create dictionary of target classes
label_dict = {
 0: "T-shirt/top",
 1: "Trouser",
 2: "Pullover",
 3: "Dress",
 4: "Coat",
 5: "Sandal",
 6: "Shirt",
 7: "Sneaker",
 8: "Bag",
 9: "Ankle boot"
}

if __name__ == "__main__":
    # in order to use the web service, first create the instance of ApiClient class
    client = ApiClient() # NOTE: you need to call ApiClient(auth_key=YOUR_API_KEY)

    # if needed, test it with echo service 
    # this is the simplest service that we can use to check if web service is running correctly
    # uncomment the code below to test echo()
    # val = client.echo("hi there!!!!")
    # print("For the echo service the returned value is: ", val)

    # you can use the method get_fashion_mnist_data to get data for fashion mnist
    # this will be rate limited to 10000 samples per call
    # for test accounts we will return a sample of 5 records to give you a feel
    num_samples = 10
    images, labels = client.get_fashion_mnist_data(num_samples)

    # you can print the labels returned by the service and examine
    # I am printing first 10 labels below
    print("Labels of the fetched Fashion MNIST data (first 5) shown below ")
    for label in labels[:5]:
        print(label, label_dict[label])

    # let us now display these images using matplotlib
    # we assume this package is already installed in your system
    for img, label in zip(images[:5], labels[:5]):
        print("y = {label_index} ({label})".format(label_index=label, label=label_dict[label]))
        plt.imshow(img, cmap='Greys')
        plt.imshow(np.array(img), cmap="gray")
        plt.show()

    # now you can continue further by vectoring the class label and creating the required dataset
    # your code ......
