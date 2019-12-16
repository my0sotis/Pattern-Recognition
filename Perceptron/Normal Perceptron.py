# Author: Zhou Pengchuan
# Time: 2019-10-8
# Python Version: 3.7.4
# IDE: PyCharm 2019.1.3

import numpy as np

if __name__ == "__main__":                                          # Entrance
    rank = int(input("Enter the rank of the matrix:"))              # Input the rank of the matrix
    num = int(input("Enter the number of the samples:"))            # Input the number of the samples

    samples = []                                                    # The list of w

    for i in range(num * 2):                                        # Input the list
        prompt = "Enter x" + str(i + 1) + " in w" + str(int(i / num) + 1) + ":"
        temp = input(prompt)
        temp = [int(n) for n in temp.split()]
        temp.append(1)                                              # Augment the matrix
        temp = np.asarray(temp)
        if int(i / num) == 1:                                       # Normalize the matrix
            temp *= -1
        samples.append(temp)

    init = input("Enter the initial matrix:")                       # Initial matrix
    init = [int(n) for n in init.split()]
    init = np.asarray(init)
    step = int(input("Enter the step:"))                            # Initial step

    time = 1                                                        # Cycles of Loop
    timeChanged = 1                                                 # Times of change
    while True:
        isChanged = False
        print("Case #" + str(time) + ":")
        time += 1
        for i in range(num * 2):                                    # Traversing the sample set
            w = init.reshape(rank, 1)
            temp = np.dot(samples[i], w)                            # Calculate the result

            if temp <= 0:
                isChanged = True
                init = init.reshape(1, rank)
                init = init + step * samples[i]                     # Change the w
                print("w" + str(timeChanged) + "Tx" + str(i + 1) + " = " + str(temp[0]) + " <= 0")
                timeChanged += 1
                print("Correction: w" + str(timeChanged) + " = w" +
                    str(timeChanged - 1) + " + x" + str(i) + " = " + str(init[0]))
                continue
            print("w" + str(timeChanged) + "Tx" + str(i + 1) + " = " + str(temp[0]) + " > 0")

        if not isChanged:                                           # if not changed break the loop
            print("Finished!")
            break
