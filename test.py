from FileLoader import FileLoader
from predict import Predict
from train import Train

if __name__ == "__main__":
    loader = FileLoader()
    data = loader.load("data.csv")

    train = Train(data)
    theta = train.fit(0.1, 1000)

    predict = Predict(data, theta)
        
    while True:
        mileage = int(input("Please enter the mileage of your car\n\t>> "))
        if mileage < 0:
            print("\t !!! invalid mileage, positive number please !!!")
            continue
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

    
 
    

