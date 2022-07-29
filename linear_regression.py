import sys
from FileLoader import FileLoader
from predict import Predict
from train import Train

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Invalid number of arguments")  
    
    loader = FileLoader()
    data = loader.load(sys.argv[1])

    # train the model to get thetas to estimate price
    train = Train(data)
    # learning rate is important as it ensures that Gradient descent converges in a reasonable time
    theta = train.gradient_descent(0.1, 1000)

    predict = Predict(data, theta)
        
    while True:
        mileage = int(input("Please enter the mileage of your car\n\t>> "))
        price = predict.predict_price(mileage)
        print(f"The estimated price of the car is {price} $")
        while True:
            print("\t- plot results of estimation typing 'v'")
            print("\t- estimate another car typing 'e'")
            print("\t- quit the program typing 'q'")
            r = input("\t>> ")
            if r == 'v':
                predict.plot(mileage, price)
                train.plot_losses()
                train.animation()
            elif r == 'e':
                break
            elif r == 'q':
                exit()

    
 
    

